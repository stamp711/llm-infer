#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cuda_bf16.h>

namespace cg = cooperative_groups;

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
    uint32_t kHeadDim;
    uint32_t kHeadDimPerThread;
    uint32_t G;  // q head per kv head; GQA factor
    uint32_t L1;
    uint32_t L2;
    uint32_t B;

    [[nodiscard]] consteval bool is_valid() const {
        return kHeadDim == kHeadDimPerThread * WARP_SIZE  // head dim is distributed across threads
               && num_threads() <= MAX_THREADS_PER_BLOCK  // make sure threads fit in SM
               && L1 % L2 == 0 && L2 % B == 0 && B >= 1 && G >= 1;
    }

    [[nodiscard]] consteval size_t block_shared_mem_kv_tile_size() const {
        return sizeof(half) * kHeadDim * L2;  // L2 tokens at a time loaded into shared memory
    }

    [[nodiscard]] consteval uint32_t num_warps_for_single_q_head_in_l1() const { return L1 / L2; }
    [[nodiscard]] consteval uint32_t num_warps_for_all_q_heads_in_l1() const {
        return G * num_warps_for_single_q_head_in_l1();
    }

    [[nodiscard]] consteval uint32_t num_warps() const { return num_warps_for_all_q_heads_in_l1(); }
    [[nodiscard]] consteval uint32_t num_threads() const { return WARP_SIZE * num_warps(); }
};

constexpr DecodingGQAPartition PMistral7B = {
    .kHeadDim = 128, .kHeadDimPerThread = 4, .G = 4, .L1 = 1024, .L2 = 128, .B = 4};

// warp processes (P.B) kv tokens for a single query head
//
// input q vector is split across threads in the warp
// input k, v is [P.B][P.kHeadDim], in shared memory
// kv_tokens is actual kv tokens in this tile, should <= P.B
//
// The return value holds:
// * m: maximum value of the dot product
// * z: normalization factor
// * o: output vector, but it's split across threads in the warp (kHeadDimPerThread) to reduce register pressure
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
        x[t] = exp(x[t] - att.m);
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
template __device__ AttentionState<PMistral7B.kHeadDimPerThread> attention_singleqhead_b<PMistral7B>(
    cg::thread_block_tile<WARP_SIZE>, float (&)[PMistral7B.kHeadDimPerThread], const half*, const half*, uint32_t);

// warp processes (P.L2) kv tokens for a single query head
//
// input q vector is split across threads in the warp
// input k, v is [P.L2][P.kHeadDim], in shared memory
// kv_tokens is actual kv tokens in this tile, should <= P.L2
//
// attention output in returned result is split across threads in the warp
//
// (P.B) tokens in (P.L2) will be processed at a time before advancing to the next (P.B) tokens,
// this is to keep register usage proportional to (P.B)
template <DecodingGQAPartition P>
    requires(P.is_valid())
__device__ AttentionState<P.kHeadDimPerThread> attention_singleqhead_l2tile(cg::thread_block_tile<WARP_SIZE> warp,
                                                                            float (&q)[P.kHeadDimPerThread],
                                                                            const half* k, const half* v,
                                                                            uint32_t kv_tokens) {
    assert(kv_tokens <= P.L2);  // runtime check in debug builds

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
template __device__ AttentionState<PMistral7B.kHeadDimPerThread> attention_singleqhead_l2tile<PMistral7B>(
    cg::thread_block_tile<WARP_SIZE>, float (&)[PMistral7B.kHeadDimPerThread], const half*, const half*, uint32_t);

// LEVEL 2 problem: warp group processes (P.L1 = 1st level tile size) kv tokens for a single query head
//
// input q vector is split across threads in the warp
// input k, v is [P.L1 (but strided)][P.kHeadDim], in global memory
// kv_tokens is actual kv tokens in this tile, should <= P.L1
//
// (P.L1) kv tokens will be divided into subproblems with size (P.L2) = 2nd level tile size, each handled by a warp
template <DecodingGQAPartition P>
    requires(P.is_valid())
__device__ void attention_single_qhead(cg::thread_block_tile<WARP_SIZE * P.L1 / P.L2> warp_group,
                                       float (&q)[P.kHeadDimPerThread], const half* k, const half* v,
                                       uint32_t kv_tokens) {
    // TODO
}

// LEVEL 1 problem: This block processes (P.L1 = 1st level tile size) kv tokens for G query heads (GQA)
//
//
// input k, v is [P.L1 (but strided)][P.kHeadDim], in global memory
// kv_tokens is actual kv tokens in this tile, should <= P.L1
//
// This will be divided into G subproblems, each subproblem solves a single q head in G for (P.L1)
template <DecodingGQAPartition P>
    requires(P.is_valid())
__device__ void attention_block(cg::thread_block block, const half* k, const half* v) {
    // TODO
}

template <DecodingGQAPartition P>
    requires(P.is_valid())
__device__ void attention_kernel(const half* k, const half* v, uint32_t kv_tokens) {
    __shared__ half k_smem[P.L2][P.kHeadDim];
    __shared__ half v_smem[P.L2][P.kHeadDim];

    auto block = cg::this_thread_block();

    auto warp_id = block.thread_index();

    // Determine this block's L1 tile
    // Grid dimensions decode which work this block does
    uint32_t kv_head_idx = blockIdx.y;  // Which of the 8 KV heads
    uint32_t l1_tile_idx = blockIdx.x;  // Which L1 tile in this KV head
    uint32_t l1_offset = l1_tile_idx * P.L1;

    // This block handles G=4 query heads that share this KV head
    uint32_t first_q_head = kv_head_idx * P.G;  // e.g., KV head 0 => Q heads 0,1,2,3

    // TODO
}
