#include <driver_types.h>

#include <cassert>
#include <cstdint>

#include "kernels.cuh"
#include "model/config.hpp"
#include "model/device.hpp"
#include "model/model.hpp"

void Model::forward_cuda_(InferenceState& s, uint32_t token, uint32_t pos, InferenceMode mode) {
    s.set_mode(mode);

    forward_cuda_build_or_update_graph_(s, token, pos, mode);

    s.graph().launch(s.stream());

    if (mode == InferenceMode::Decode) {
        CUDAContext::synchronize_stream(s.stream());
        CUDAContext::check_last_error();
    }
}

void Model::forward_cuda_build_or_update_graph_(InferenceState& s, uint32_t token, uint32_t pos, InferenceMode mode) {
    const ModelConfig& c = *config_;

    // Copy embedding
    {
        uint32_t block_size = 256;

        cudaKernelNodeParams params{
            .gridDim = {1, 1, 1},  // Will be set based on quantization type
            .blockDim = {block_size, 1, 1},
        };

        switch (token_embedding_table_.quantization()) {
            case QuantizationType::FP32: {
                params.func = reinterpret_cast<void*>(copy_embedding_float);
                uint32_t n_vecs = c.dim / 4;  // Each thread processes float4
                params.gridDim = {(n_vecs + block_size - 1) / block_size, 1, 1};
                break;
            }
            case QuantizationType::FP16: {
                params.func = reinterpret_cast<void*>(copy_embedding_half);
                uint32_t n_vecs = c.dim / 2;  // Each thread processes half2
                params.gridDim = {(n_vecs + block_size - 1) / block_size, 1, 1};
                break;
            }
            default: throw std::runtime_error("Unsupported embedding table quantization type");
        }

        s.graph().add_or_update_kernel_node("copy_embedding", params,
                                            KernelArgs{s.x(), token_embedding_table_.data(), token, c.dim});
    }

    // StreamingLLM
    uint32_t kv_sink = pos >= c.max_seq_len ? KV_SINK : 0;
    uint32_t kv_pos = kv_sink + ((pos - kv_sink) % (c.max_seq_len - kv_sink));
    uint32_t kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;

    // Let each block add or update kernel nodes
    for (auto& b : blocks_) {
        b->block(s, pos, kv_sink, kv_pos, kv_len);
    }

    // Final layer norm, s.x -> s.x
    switch (c.norm_type) {
        case LayerNormType::RMSNorm: {
            switch (c.norms_weight_quantization) {
                case QuantizationType::FP32: {
                    cudaKernelNodeParams params{
                        .func = reinterpret_cast<void*>(rmsnorm),
                        .gridDim = {1, 1, 1},
                        .blockDim = {128, 1, 1},
                    };
                    s.graph().add_or_update_kernel_node(
                        "rmsnorm", params,
                        KernelArgs{s.x(), s.x(), static_cast<const float*>(rms_final_weight_.data()), c.dim,
                                   c.norm_eps});
                } break;
                default: throw std::runtime_error("Unsupported layer norm weight quantization");
            }
            break;
        }
        default: throw std::runtime_error("Unsupported layer norm type");
    }
}
