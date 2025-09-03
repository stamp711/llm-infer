#include <driver_types.h>

#include <cstddef>
#include <optional>

#include "kernels.cuh"
#include "model/block.hpp"
#include "model/config.hpp"
#include "model/device.hpp"

// Explicit instantiation for f16_t weights with float normalization
template void Block::block_cuda_<std::uint16_t, float>(InferenceState& s, uint32_t pos, uint32_t kv_sink,
                                                       uint32_t kv_pos, uint32_t kv_len);

template <typename WeightT, typename NormT>
void Block::block_cuda_(InferenceState& s, uint32_t pos, uint32_t kv_sink, uint32_t kv_pos, uint32_t kv_len) {
    const ModelConfig& c = *config;

    // For now, validate norm is RMSNorm, weight is FP16, norm weight is FP32
    if (c.norm_type != LayerNormType::RMSNorm || c.weight_quantization != QuantizationType::FP16 ||
        c.norms_weight_quantization != QuantizationType::FP32) {
        throw std::invalid_argument("Invalid configuration for block_cuda");
    }

    // Attention pre-norm, s.x -> s.xb
    std::string att_pre_norm = fmt::format("{}:pre-attn-rmsnorm", layer_i);
    {
        s.graph().add_or_update_kernel_node(att_pre_norm,
                                            cudaKernelNodeParams{
                                                .func = reinterpret_cast<void*>(rmsnorm),
                                                .gridDim = {1, 1, 1},
                                                .blockDim = {128, 1, 1},
                                            },
                                            KernelArgs{s.xb(), s.x(), rms_att_weight_.data(), c.dim, c.norm_eps});
    }

    uint32_t q_total_dim = c.n_heads * c.head_dim;      // all heads' q combined dim
    uint32_t kv_total_dim = c.n_kv_heads * c.head_dim;  // all kv-heads' k/v combined dim

    // QK + Clip, s.xb -> s.{q,k}; V + Clip, s.xb -> v_cache
    // Separate concurrent kernels.
    std::string calc_q = fmt::format("{}:q+clip", layer_i);
    std::string calc_k = fmt::format("{}:k+clip", layer_i);
    std::string calc_v = fmt::format("{}:v+clip", layer_i);
    {
        {
            constexpr uint32_t block_size = 128;
            uint32_t gemv_warps_needed = q_total_dim;  // Each warp calculates a single q_i
            uint32_t threads_needed = gemv_warps_needed * CUDAContext::get().warp_size();
            uint32_t grid_size = (threads_needed + block_size - 1) / block_size;

            s.graph().add_or_update_kernel_node(calc_q, KernelDeps{att_pre_norm},
                                                cudaKernelNodeParams{
                                                    .func = reinterpret_cast<void*>(fused_gemv_clip),
                                                    .gridDim = {grid_size, 1, 1},    // TODO
                                                    .blockDim = {block_size, 1, 1},  // TODO
                                                },
                                                KernelArgs{
                                                    s.q(),
                                                    s.xb(),
                                                    wq.data(),
                                                    c.dim,
                                                    q_total_dim,
                                                    c.qkv_clip ? std::make_optional(c.qkv_clip_value) : std::nullopt,
                                                });
        }

        {
            constexpr uint32_t block_size = 128;
            uint32_t gemv_warps_needed = kv_total_dim;  // Each warp calculates a single q_i
            uint32_t threads_needed = gemv_warps_needed * CUDAContext::get().warp_size();
            uint32_t grid_size = (threads_needed + block_size - 1) / block_size;

            s.graph().add_or_update_kernel_node(calc_k, KernelDeps{att_pre_norm},
                                                cudaKernelNodeParams{
                                                    .func = reinterpret_cast<void*>(fused_gemv_clip),
                                                    .gridDim = {grid_size, 1, 1},
                                                    .blockDim = {block_size, 1, 1},
                                                },
                                                KernelArgs{
                                                    s.k(),
                                                    s.xb(),
                                                    wk.data(),
                                                    c.dim,
                                                    kv_total_dim,
                                                    c.qkv_clip ? std::make_optional(c.qkv_clip_value) : std::nullopt,
                                                });
        }

        {
            constexpr uint32_t block_size = 128;
            uint32_t gemv_warps_needed = kv_total_dim;  // Each warp calculates a single q_i
            uint32_t threads_needed = gemv_warps_needed * CUDAContext::get().warp_size();
            uint32_t grid_size = (threads_needed + block_size - 1) / block_size;

            s.graph().add_or_update_kernel_node(
                calc_v, KernelDeps{att_pre_norm},
                cudaKernelNodeParams{
                    .func = reinterpret_cast<void*>(fused_gemv_clip_f2h),
                    .gridDim = {grid_size, 1, 1},
                    .blockDim = {block_size, 1, 1},
                },
                KernelArgs{
                    v_cache.data() + static_cast<size_t>(kv_pos * kv_total_dim),  // directly insert to V cache
                    s.xb(),
                    wv.data(),
                    c.dim,
                    kv_total_dim,
                    c.qkv_clip ? std::make_optional(c.qkv_clip_value) : std::nullopt,
                });
        }
    }

    // RoPE for q, s.q -> s.q; RoPE + cache insert for k, s.k -> k_cache
    std::string rope_q_node = fmt::format("{}:rope_q", layer_i);
    std::string rope_k_insert_cache_node = fmt::format("{}:rope_k_insert_cache", layer_i);
    {
        {
            constexpr uint32_t block_size = 128;
            uint32_t elems_to_process = q_total_dim;
            uint32_t threads_needed = elems_to_process / 2;  // Each thread process 2 whole elems
            uint32_t grid_size = (threads_needed + block_size - 1) / block_size;

            s.graph().add_or_update_kernel_node(rope_q_node, KernelDeps{calc_q},
                                                cudaKernelNodeParams{
                                                    .func = reinterpret_cast<void*>(rope_f2f),
                                                    .gridDim = {grid_size, 1, 1},
                                                    .blockDim = {block_size, 1, 1},
                                                },
                                                KernelArgs{
                                                    s.q(),
                                                    s.q(),
                                                    c.n_heads,
                                                    c.head_dim,
                                                    pos,
                                                    c.rope_theta,
                                                    c.rotary_dim,
                                                });
        }

        {
            constexpr uint32_t block_size = 128;
            uint32_t elems_to_process = kv_total_dim;
            uint32_t threads_needed = elems_to_process / 2;  // Each thread process 2 whole elems
            uint32_t grid_size = (threads_needed + block_size - 1) / block_size;

            s.graph().add_or_update_kernel_node(rope_k_insert_cache_node, KernelDeps{calc_k},
                                                cudaKernelNodeParams{
                                                    .func = reinterpret_cast<void*>(rope_f2h),
                                                    .gridDim = {grid_size, 1, 1},
                                                    .blockDim = {block_size, 1, 1},
                                                },
                                                KernelArgs{
                                                    k_cache.data() + static_cast<size_t>(kv_pos * kv_total_dim),
                                                    s.k(),
                                                    c.n_kv_heads,
                                                    c.head_dim,
                                                    pos,
                                                    c.rope_theta,
                                                    c.rotary_dim,
                                                });
        }
    }

    // Sink token k rotation
    // TODO: better be fused in attention, or we just store non-RoPE-ed caches.
    std::string sink_token_rotation = fmt::format("{}:sink_token_rotation", layer_i);
    {
        uint32_t block_size = 128;
        uint32_t elems_to_process = kv_total_dim * kv_sink;  // We combine all sink tokens.

        uint32_t threads_needed = elems_to_process / 2;  // Each thread process 2 whole elems
        uint32_t grid_size = (threads_needed + block_size - 1) / block_size;

        if (kv_sink == 0) {
            // NOTE: when kv_sink = 0, we still launch a single grid, since launching with 0 dim is undefined behaviour.
            // n_heads will be 0 in args, so all launched threads will be masked out by index check, doing nothing.
            // Not even a single memory access will happen.
            block_size = 1;
            grid_size = 1;
        }

        s.graph().add_or_update_kernel_node(
            sink_token_rotation,
            KernelDeps{},  // We don't need to depend on the K cache update above because:
                           // 1. When KV cache window is not shifted, this kernel will do nothing
                           //    (not even a single memory access will happen).
                           // 2. When KV cache window is already shifted, the sink tokens will not be modified, the
                           // above kernel will not write to the same position as us.
            cudaKernelNodeParams{
                .func = reinterpret_cast<void*>(rope_f2f),
                .gridDim = {grid_size, 1, 1},
                .blockDim = {block_size, 1, 1},
            },
            KernelArgs{
                k_cache.data(),  // Sink tokens reside at beginning of the cache
                k_cache.data(),
                c.n_kv_heads * kv_sink,  // We are combining sink tokens, and every head_dim is processed similarly, so
                                         // just pretend there's more heads
                c.head_dim,
                1,  // Rotate forward by 1
                c.rope_theta,
                c.rotary_dim,
            });
    }

    // GQA in FlashAttention style
    // TODO: if we use FP16 for x, we could potentially leverage tensor cores? (MMA) > but need to duplicate K blocks in
    // shared memory, maybe not worth it.
    {
        constexpr uint32_t block_size = 128;
        uint32_t elems_to_process = kv_total_dim;
        uint32_t threads_needed = elems_to_process / 2;  // Each thread process 2 whole elems
        uint32_t grid_size = (threads_needed + block_size - 1) / block_size;

        s.graph().add_or_update_kernel_node(rope_k_insert_cache_node, KernelDeps{calc_k},
                                            cudaKernelNodeParams{
                                                .func = reinterpret_cast<void*>(rope_f2h),
                                                .gridDim = {grid_size, 1, 1},
                                                .blockDim = {block_size, 1, 1},
                                            },
                                            KernelArgs{
                                                k_cache.data() + static_cast<size_t>(kv_pos * kv_total_dim),
                                                s.k(),
                                                c.n_kv_heads,
                                                c.head_dim,
                                                pos,
                                                c.rope_theta,
                                                c.rotary_dim,
                                            });
    }

    // Transform back to [dim] and residual connection

    // FFN pre-norm

    // FFN + residual connection
}
