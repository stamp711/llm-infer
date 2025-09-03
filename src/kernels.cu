#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <optional>

constexpr uint32_t FULL_MASK = 0xFFFFFFFF;

// Helper device functions
namespace {
__device__ inline float warp_reduce_sum(float val) {
#pragma unroll
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(FULL_MASK, val, offset);
    }
    return val;
}

__device__ inline float block_reduce_sum(float val) {
    // Each warp reduce first
    val = warp_reduce_sum(val);

    // If threads in block is less than warp size, warp reduction alone is enough
    if (blockDim.x < warpSize) {
        return val;
    }

    // need to <= warpSize, or we need multiple reduction steps. Fortunately NVIDIA GPUs currently guarantee this.
    constexpr uint32_t MAX_WARPS_PER_BLOCK = 1024 / 32;  // 32
    __shared__ float warp_sums[MAX_WARPS_PER_BLOCK];

    const uint32_t tid = threadIdx.x;
    const uint32_t warp_id = threadIdx.x / warpSize;
    const uint32_t lane_id = threadIdx.x % warpSize;
    const uint32_t n_warps = blockDim.x / warpSize;

    // First thread in each wrap writes to warp_sums
    if (lane_id == 0) warp_sums[warp_id] = val;
    __syncthreads();

    // Final reduction on first warp, store result to warp_sums[0]
    if (warp_id == 0) {
        val = tid < n_warps ? warp_sums[tid] : 0.F;
        val = warp_reduce_sum(val);

        if (tid == 0) {
            warp_sums[0] = val;
        }
    }
    __syncthreads();

    return warp_sums[0];
}

__device__ float dot_product(const float* x, const half* y, uint32_t size) {
    float sum = 0;
    for (uint32_t i = threadIdx.x % warpSize; i < size; i += warpSize) {
        float v = __half2float(y[i]) * x[i];
        sum += v;
    }
    return warp_reduce_sum(sum);
}

__device__ inline float2 rope(float2 a, uint32_t i, uint32_t pos, float theta, uint32_t rotary_dim) {
    if (i + 1 < rotary_dim) {
        float freq = 1.F / powf(theta, static_cast<float>(i) / static_cast<float>(rotary_dim));
        float deg = static_cast<float>(pos) * freq;
        float cos = std::cosf(deg);
        float sin = std::sinf(deg);

        float2 res = {(a.x * cos) - (a.y * sin), (a.x * sin) + (a.y * cos)};
        return res;
    }
    return a;
}

}  // namespace

// Templates
namespace {

template <typename OutType, typename InType>
__device__ void rope_template(OutType* out, const InType* x, uint32_t n_heads, uint32_t head_dim, uint32_t pos,
                              float theta, uint32_t rotary_dim) {
    uint32_t tid = (blockIdx.x * blockDim.x) + threadIdx.x;

    // Each thread handles 2 consecutive elements
    // NOTE: if rotary_dim is much smaller than head_dim, we should skip unrotated dims when calculating pari_idx (but
    // still ensure coalesced memory access in each warp)
    uint32_t pair_first_idx = tid * 2;

    if (pair_first_idx + 1 < n_heads * head_dim) {
        uint32_t inner_idx = pair_first_idx % head_dim;

        // Input type conversion to float2
        float2 pair;
        if constexpr (std::is_same_v<InType, float>) {
            pair = reinterpret_cast<const float2*>(x)[tid];
        } else if constexpr (std::is_same_v<InType, half>) {
            half2 h2_pair = reinterpret_cast<const half2*>(x)[tid];
            pair = __half22float2(h2_pair);
        }

        float2 rotated = rope(pair, inner_idx, pos, theta, rotary_dim);

        // Output type conversion
        if constexpr (std::is_same_v<OutType, float>) {
            reinterpret_cast<float2*>(out)[tid] = rotated;
        } else if constexpr (std::is_same_v<OutType, half>) {
            reinterpret_cast<half2*>(out)[tid] = __float22half2_rn(rotated);
        }
    }
}

template <typename OutType, typename Epilogue>
__device__ void fused_gemv_(OutType* out, const float* x /* 1xn */, const half* w /* nxd */, uint32_t n, uint32_t d,
                            Epilogue epilogue) {
    uint32_t warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    if (warp_id >= d) return;

    // Each warp cooperate to calculate a single value in the output array
    const half* ww = w + (static_cast<size_t>(warp_id * n));  // warp_id -th vector in w
    float res = dot_product(x, ww, n);

    // epilogue
    res = epilogue(res);

    // type conversion
    if constexpr (std::is_same_v<OutType, float>) {
        out[warp_id] = res;
    } else if constexpr (std::is_same_v<OutType, half>) {
        out[warp_id] = __float2half(res);
    }
}

}  // namespace

// ====================================
// ========== Copy Embedding ==========
// ====================================

__global__ void copy_embedding_float(float* out, const float* embedding_table, uint32_t token, uint32_t dim) {
    uint32_t tid = threadIdx.x + (blockIdx.x * blockDim.x);  // 1D
    const float* src = embedding_table + (static_cast<size_t>(token * dim));

    if (tid * 4 < dim) {
        const auto* srcv = reinterpret_cast<const float4*>(src);
        auto* outv = reinterpret_cast<float4*>(out);
        outv[tid] = srcv[tid];
    }
}

__global__ void copy_embedding_half(float* out, const half* token_embedding_table, uint32_t token, uint32_t dim) {
    uint32_t tid = threadIdx.x + (blockIdx.x * blockDim.x);  // 1D

    const half* src = token_embedding_table + (static_cast<size_t>(token * dim));

    uint32_t idx = tid * 2;
    if (idx < dim) {
        const auto* srcv = reinterpret_cast<const half2*>(src);
        half2 h2 = srcv[tid];
        out[idx] = __half2float(h2.x);
        out[idx + 1] = __half2float(h2.y);
    }
}

// ==============================
// ========== RMS Norm ==========
// ==============================

__global__ void rmsnorm(float* out, const float* x, const float* weight, uint32_t size, float eps) {
    // only a single 1D block is launched so we can use block reduction.
    // Cross-SM reduction is only available on Hopper. (We are targeting Ada Lovelace)

    const auto* xv = reinterpret_cast<const float4*>(x);
    const auto* weightv = reinterpret_cast<const float4*>(weight);
    auto* outv = reinterpret_cast<float4*>(out);

    const uint32_t tid = threadIdx.x;
    const uint32_t n_threads = blockDim.x;
    const uint32_t vecs_total = size / 4;
    const uint32_t vecs_per_thread = vecs_total / blockDim.x;

    float rms_sum = 0;
    for (uint32_t i = 0; i < vecs_per_thread; ++i) {
        uint32_t id = (i * n_threads) + tid;
        if (id < vecs_total) {
            float4 v = xv[id];
            rms_sum += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }

    rms_sum = block_reduce_sum(rms_sum);
    float scale = rsqrtf((rms_sum / static_cast<float>(size)) + eps);

    for (uint32_t i = 0; i < vecs_per_thread; ++i) {
        uint32_t id = (i * n_threads) + tid;
        if (id < vecs_total) {
            float4 v = xv[id];
            float4 w = weightv[id];

            v.x = v.x * scale * w.x;
            v.y = v.y * scale * w.y;
            v.z = v.z * scale * w.z;
            v.w = v.w * scale * w.w;

            outv[id] = v;
        }
    }
}

// =======================================
// ========== fused GEMV + Clip ==========
// =======================================

struct ClipEpilogue {
    std::optional<float> clip;

    __device__ inline float operator()(float x) const {
        if (clip.has_value()) {
            return min(x, clip.value());
        }
        return x;
    }
};

__global__ void fused_gemv_clip(float* out, const float* x /* 1xn */, const half* w /* nxd */, uint32_t n, uint32_t d,
                                std::optional<float> clip) {
    ClipEpilogue epilogue{clip};
    fused_gemv_<float>(out, x, w, n, d, epilogue);
}

__global__ void fused_gemv_clip_f2h(half* out, const float* x /* 1xn */, const half* w /* nxd */, uint32_t n,
                                    uint32_t d, std::optional<float> clip) {
    ClipEpilogue epilogue{clip};
    fused_gemv_<half>(out, x, w, n, d, epilogue);
}

// ==========================
// ========== RoPE ==========
// ==========================

__global__ void rope_f2f(float* out, const float* x, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta,
                         uint32_t rotary_dim) {
    rope_template<float, float>(out, x, n_heads, head_dim, pos, theta, rotary_dim);
}

__global__ void rope_f2h(half* out, const float* x, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta,
                         uint32_t rotary_dim) {
    rope_template<half, float>(out, x, n_heads, head_dim, pos, theta, rotary_dim);
}

__global__ void rope_h2h(half* out, const half* x, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta,
                         uint32_t rotary_dim) {
    rope_template<half, half>(out, x, n_heads, head_dim, pos, theta, rotary_dim);
}

// ===============================
// ========== Attention ==========
// ===============================

template <uint32_t N, uint32_t B>
__device__ inline void attn_block(float* x /* 1xn */, half* k /* nxb */, half* v /* nxb */, uint32_t n /* head dim */,
                                  uint32_t b /* kv tile length */) {
    __shared__ float scores[B];

    for (uint32_t i = 0; i < B; ++i) {
        float sum = 0;
        for (uint32_t j = threadIdx.x; j < N; j += blockDim.x) {
            // sum += x[j] * k[j];
        }
        scores[i] = sum;
    }
}
