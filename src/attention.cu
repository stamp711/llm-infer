#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda_bf16.h>

#include <cstddef>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

// SoA helper
template <int D>
struct AttentionStateRef {
    float& m;       // NOLINT
    float& z;       // NOLINT
    float (&o)[D];  // NOLINT

    __device__ void merge(AttentionStateRef b) {
        auto m_c = fmaxf(m, b.m);

        auto eac = m == m_c ? 1.0F : expf(m - m_c);
        auto ebc = b.m == m_c ? 1.0F : expf(b.m - m_c);

        auto z_c = (z * eac) + (b.z * ebc);

        float oas = (z / z_c) * eac;
        float obs = (b.z / z_c) * ebc;

        for (int i = 0; i < D; ++i) {
            o[i] = o[i] * oas + b.o[i] * obs;
        }

        m = m_c;
        z = z_c;
    }
};

// * m: maximum value of the dot product
// * z: normalization factor
// * o: output vector, but it may be split across threads in the warp (kHeadDimPerThread) to reduce register pressure
template <int D>
struct AttentionState {
    float m = -INFINITY;
    float z = 0.0F;
    float o[D] = {};

    __device__ void merge(const AttentionState& b) {
        auto m_c = fmaxf(m, b.m);

        auto eac = m == m_c ? 1.0F : expf(m - m_c);
        auto ebc = b.m == m_c ? 1.0F : expf(b.m - m_c);

        auto z_c = (z * eac) + (b.z * ebc);

        float oas = (z / z_c) * eac;
        float obs = (b.z / z_c) * ebc;

        for (int i = 0; i < D; ++i) {
            o[i] = o[i] * oas + b.o[i] * obs;
        }

        m = m_c;
        z = z_c;
    }
};

const uint32_t WARP_SIZE = 32;
const uint32_t MAX_THREADS_PER_BLOCK = 1024;

struct DecodingGQAPartition {
    uint32_t kKvHeads;
    uint32_t kHeadDim;
    uint32_t kHeadDimPerThread;
    uint32_t G;   // q head per kv head; GQA factor
    uint32_t L1;  // block tile; a block handles L1 (better be made dynamic)
    uint32_t L2;  // smem tile; load L2 sized tile into block shared memory at a time (can be made dynamic through
                  // extern __shared__ memory)
    uint32_t L3;  // warp tile; within L2, each (L3 tile + single q head) work is processed in parallel, each by a
                  // single warp
    uint32_t B;   // register tile; within L3, warp processes B tokens in each iteration
    uint32_t kMaxSeqLen;

    [[nodiscard]] consteval bool is_valid() const {
        return kHeadDim == kHeadDimPerThread * WARP_SIZE  // head dim is distributed across threads
               && num_threads() <= MAX_THREADS_PER_BLOCK  // make sure threads fit in SM
               && L1 % L2 == 0 && L2 % L3 == 0 && L3 % B == 0 && B >= 1 && G >= 1;
    }

    [[nodiscard]] uint32_t this_thread_offset_in_head_dim() const {
        return kHeadDimPerThread * (threadIdx.x % warpSize);
    }

    [[nodiscard]] consteval size_t block_shared_mem_kv_tile_size() const {
        return sizeof(half) * kHeadDim * L2;  // L2 tokens at a time loaded into shared memory
    }

    [[nodiscard]] consteval uint32_t num_warps() const { return L2 / L3 * G; }
    [[nodiscard]] consteval uint32_t num_threads() const { return WARP_SIZE * L2 / L3 * G; }

    [[nodiscard]] consteval uint32_t num_warps_for_single_l3() const { return L2 / L3; }
    [[nodiscard]] consteval uint32_t num_threads_for_single_l3() const { return WARP_SIZE * L2 / L3; }
};

constexpr DecodingGQAPartition PMistral7B = {.kKvHeads = 8,
                                             .kHeadDim = 128,
                                             .kHeadDimPerThread = 4,
                                             .G = 4,
                                             .L1 = 512,
                                             .L2 = 32,
                                             .L3 = 16,
                                             .B = 4,
                                             .kMaxSeqLen = 4096};

// warp processes (P.B) kv tokens for a single query head
//
// input q vector is split across threads in the warp
// input k, v is [P.B][P.kHeadDim], in shared memory
// kv_tokens is actual kv tokens in this tile, should <= P.B
//
// This uses registers proportionally to (P.B)
template <DecodingGQAPartition P>
    requires(P.is_valid())
__device__ AttentionState<P.kHeadDimPerThread> attention_singleqhead_b(cg::thread_block_tile<WARP_SIZE> warp,
                                                                       float (&q)[P.kHeadDimPerThread], const half* k,
                                                                       const half* v, uint32_t kv_tokens) {
    assert(kv_tokens <= P.B);  // runtime check in debug builds

    uint32_t this_thread_offset_in_head_dim = P.kHeadDimPerThread * warp.thread_rank();

    float x[P.B] = {};

    for (uint32_t t = 0; t < kv_tokens; ++t) {
        const auto* kk = k + (static_cast<size_t>(t * P.kHeadDim)) + this_thread_offset_in_head_dim;
        for (uint32_t i = 0; i < P.kHeadDimPerThread; ++i) {
            // calculate partial dot product on this kHeadDimPerThread
            x[t] += q[i] * __half2float(kk[i]);
        }
    }

    AttentionState<P.kHeadDimPerThread> att;

    // Threads in this warp reduce to get full dot products, apply scaling, also determine max value
    for (uint32_t t = 0; t < kv_tokens; ++t) {
        // reduction of t-th full dot product
        x[t] = cg::reduce(warp, x[t], cg::plus<float>());
        x[t] *= rsqrtf(P.kHeadDim);
        att.m = fmaxf(att.m, x[t]);
    }

    // calculate denominator for softmax
    for (uint32_t t = 0; t < kv_tokens; ++t) {
        x[t] = expf(x[t] - att.m);
        att.z += x[t];
    }

    for (uint32_t t = 0; t < kv_tokens; ++t) {
        // v projection on this kHeadDimPerThread
        const auto* vv = v + (static_cast<size_t>(t * P.kHeadDim)) + this_thread_offset_in_head_dim;

        // softmax for this x
        float s = x[t] / att.z;

        // attention output projection for this kHeadDimPerThread
        for (uint32_t i = 0; i < P.kHeadDimPerThread; ++i) {
            att.o[i] += s * __half2float(vv[i]);
        }
    }

    return att;
}

// warp processes (P.L3) kv tokens for a single query head
//
// input q vector is split across threads in the warp
// input k, v is [P.L3][P.kHeadDim], in shared memory
// kv_tokens is actual kv tokens in this tile, should <= P.L3
//
// attention output in returned result is split across threads in the warp
//
// (P.B) tokens in (P.L3) will be processed at a time before advancing to the next (P.B) tokens,
// this is to keep register usage proportional to (P.B)
template <DecodingGQAPartition P>
    requires(P.is_valid())
__device__ AttentionState<P.kHeadDimPerThread> attention_singleqhead_l3tile(cg::thread_block_tile<WARP_SIZE> warp,
                                                                            float (&q)[P.kHeadDimPerThread],
                                                                            const half* k, const half* v,
                                                                            uint32_t kv_tokens) {
    assert(kv_tokens <= P.L3);  // runtime check in debug builds

    AttentionState<P.kHeadDimPerThread> att;

    for (uint32_t t = 0; t < kv_tokens; t += P.B) {
        uint32_t kv_tokens_b = min(P.B, kv_tokens - t);  // sub problem size
        const half* kb = k + (static_cast<size_t>(t * P.kHeadDim));
        const half* vb = v + (static_cast<size_t>(t * P.kHeadDim));
        auto att_b = attention_singleqhead_b<P>(warp, q, kb, vb, kv_tokens_b);
        att.merge(att_b);
    }

    return att;
}

// Process entire L2 tile for a query head (P.L2 / P.L3 warps, in parallel)
template <DecodingGQAPartition P>
__device__ AttentionState<P.kHeadDimPerThread> attention_singleqhead_l2tile(
    cg::thread_block_tile<WARP_SIZE * P.L2 / P.L3> qhead_group,  // P.L2 / P.L3 warps
    uint32_t qhead_idx,                                          // which q head (of G q heads) for this qhead_group
    float (&q)[P.kHeadDimPerThread],
    const half* k_smem,  // L2 tile in shared memory
    const half* v_smem,  // L2 tile in shared memory
    uint32_t kv_tokens) {
    auto warp = cg::tiled_partition<WARP_SIZE>(qhead_group);

    uint32_t warp_in_qhead = qhead_group.thread_rank() / WARP_SIZE;
    uint32_t this_thread_offset_in_head_dim = P.kHeadDimPerThread * warp.thread_rank();

    // Each warp processes its L3 portion
    uint32_t l3_offset = warp_in_qhead * P.L3;
    uint32_t l3_tokens = min(P.L3, kv_tokens - l3_offset);

    auto att = attention_singleqhead_l3tile<P>(warp, q, k_smem + static_cast<size_t>(l3_offset * P.kHeadDim),
                                               v_smem + static_cast<size_t>(l3_offset * P.kHeadDim), l3_tokens);

    // Reduce across warps (needs shared memory for reduction)
    __shared__ float m[P.G][P.L2 / P.L3];
    __shared__ float z[P.G][P.L2 / P.L3];
    __shared__ float o[P.G][P.L2 / P.L3][P.kHeadDim];

    // first thread in each warp write m and z
    if (warp.thread_rank() == 0) {
        m[qhead_idx][warp_in_qhead] = att.m;
        z[qhead_idx][warp_in_qhead] = att.z;
    }

    // every thread write its own projection of o
    for (uint32_t i = 0; i < P.kHeadDimPerThread; i++) {
        o[qhead_idx][warp_in_qhead][this_thread_offset_in_head_dim + i] = att.o[i];
    }

    qhead_group.sync();

    // First thread of first warp reduces
    // TODO: could use parallel tree reduction
    if (qhead_group.thread_rank() == 0) {
        AttentionStateRef<P.kHeadDim> att_ref{
            .m = m[qhead_idx][0],
            .z = z[qhead_idx][0],
            .o = o[qhead_idx][0],
        };
        for (uint32_t i = 1; i < P.L2 / P.L3; i++) {
            AttentionStateRef<P.kHeadDim> att_ref_i{
                .m = m[qhead_idx][i],
                .z = z[qhead_idx][i],
                .o = o[qhead_idx][i],
            };
            att_ref.merge(att_ref_i);
        }
    }

    qhead_group.sync();

    // All threads read the final merged result
    AttentionState<P.kHeadDimPerThread> result;
    result.m = m[qhead_idx][0];  // Broadcast m
    result.z = z[qhead_idx][0];  // Broadcast z

    // Each thread reads its portion of the merged o
    for (uint32_t i = 0; i < P.kHeadDimPerThread; i++) {
        result.o[i] = o[qhead_idx][0][this_thread_offset_in_head_dim + i];
    }

    return result;
}

// Block processes (P.L1 = 1st level tile size) kv tokens in a single kv head'a KV cache, for G query heads
//
// input q is [P.kKvHeads][P.G][P.kHeadDim], in global memory
//
// input k, v is [P.kKvHeads][P.kMaxSeqLen][P.kHeadDim], in global memory.
// TODO: this is new layout, need to change other kernels.
//
// o is the output buffer [P.kKvHeads][P.G][P.kHeadDim], in global memory. It could be the same as q
//
template <DecodingGQAPartition P>
    requires(P.is_valid())
__global__ void attention_kernel(float* o, const float* q, const half* k, const half* v, uint32_t kv_tokens) {
    auto block = cg::this_thread_block();

    assert(block.size() == P.num_threads());

    // ===== Double Buffer =====
    constexpr uint32_t stages = 2;
    __shared__ half k_cache[stages][P.L2][P.kHeadDim];
    __shared__ half v_cache[stages][P.L2][P.kHeadDim];

    // Thread 0 manages the double buffer pipeline
    cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();

    // Determine this block's L1 tile
    uint32_t block_kv_head_idx = blockIdx.y;  // Which of the kKvHeads
    uint32_t block_l1_tile_idx = blockIdx.x;  // Which L1 tile in this KV head

    // Partition block into G groups, one per query head, then into (L2/L3) subgroups
    // uint32_t warps = P.G * P.L2 / P.L3;
    uint32_t warp_id = threadIdx.x / WARP_SIZE;
    uint32_t my_warp_qhead_idx = warp_id / (P.L2 / P.L3);
    // uint32_t my_warp_l3_tile_idx = warp_id % (P.L2 / P.L3);

    // Each thread loads its portion of q into register
    float q_local[P.kHeadDimPerThread];
    const float* block_q_start = q + static_cast<size_t>(P.kHeadDim * P.G * block_kv_head_idx);
    const float* my_warp_q_start = block_q_start + static_cast<size_t>(my_warp_qhead_idx * P.kHeadDim);
    uint32_t thread_offset_in_head_dim = P.kHeadDimPerThread * (threadIdx.x % WARP_SIZE);
    for (uint32_t i = 0; i < P.kHeadDimPerThread; i++) {
        q_local[i] = my_warp_q_start[thread_offset_in_head_dim + i];
    }

    AttentionState<P.kHeadDimPerThread> att;

    const uint32_t block_kv_token_start = block_l1_tile_idx * P.L1;
    const uint32_t block_kv_tokens = min(P.L1, kv_tokens - block_kv_token_start);
    const uint32_t num_l2_tiles = (block_kv_tokens + P.L2 - 1) / P.L2;
    const size_t l1_offset_in_kv =
        static_cast<size_t>(P.kHeadDim) *
        (block_kv_head_idx * P.kMaxSeqLen /* seek to this kv head */ + block_kv_token_start /* seek to start of L1 */);

    for (uint32_t compute_idx = 0, fetch_idx = 0; compute_idx < num_l2_tiles; ++compute_idx) {
        // Fill prefetch pipeline
        for (; fetch_idx < num_l2_tiles && fetch_idx < (compute_idx + stages); ++fetch_idx) {
            if (threadIdx.x == 0) {
                uint32_t buffer_idx = fetch_idx % stages;
                uint32_t l2_offset = fetch_idx * P.L2;
                uint32_t l2_tokens = min(P.L2, block_kv_tokens - l2_offset);
                size_t copy_size = sizeof(half) * P.kHeadDim * l2_tokens;
                uint32_t l2_offset_in_kv = l2_offset * P.kHeadDim;

                pipeline.producer_acquire();
                cuda::memcpy_async(&k_cache[buffer_idx][0][0], k + l1_offset_in_kv + l2_offset_in_kv, copy_size,
                                   pipeline);
                cuda::memcpy_async(&v_cache[buffer_idx][0][0], v + l1_offset_in_kv + l2_offset_in_kv, copy_size,
                                   pipeline);
                pipeline.producer_commit();
            }
        }

        // Wait for current compute tile to be ready
        if (threadIdx.x == 0) {
            pipeline.consumer_wait();
        }
        block.sync();

        // Process current tile
        uint32_t buffer_idx = compute_idx % stages;
        uint32_t l2_offset = compute_idx * P.L2;
        uint32_t l2_tokens = min(P.L2, block_kv_tokens - l2_offset);

        constexpr uint32_t threads_per_qhead = WARP_SIZE * (P.L2 / P.L3);
        auto qhead_group = cg::tiled_partition<threads_per_qhead>(block);

        auto att_l2 = attention_singleqhead_l2tile<P>(
            qhead_group, my_warp_qhead_idx, q_local, &k_cache[buffer_idx][0][0], &v_cache[buffer_idx][0][0], l2_tokens);
        att.merge(att_l2);

        // Release current tile after all threads finished this L2
        block.sync();
        if (threadIdx.x == 0) pipeline.consumer_release();
    }

    // TODO: cross-block reduce
}
template __global__ void attention_kernel<PMistral7B>(float* o, const float* q, const half* k, const half* v,
                                                      uint32_t kv_tokens);
