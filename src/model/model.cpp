#include "model/model.hpp"

#include <cstddef>

#include "kernels_cpu.hpp"
#include "model/block.hpp"
#include "model/config.hpp"
#include "model/device.hpp"
#include "model/inference_state.hpp"

Model::~Model() {
    if (device_type_ == DeviceType::CUDA) {
        CUDAContext::synchronize();  // TODO: move this to the outside. Destructors are noexcept by default.
    }
}

Model::Model(const GGUF& gguf, DeviceType device_type) : device_type_(device_type) {
    config_ = std::make_unique<ModelConfig>(gguf);

    if (device_type == DeviceType::CUDA) {
        CUDAContext::initialize();  // TODO: move this to the outside
    }

    // Load global tensors with shape and quantization checking
    std::string token_embd_name = "token_embd.weight";
    const auto& token_embd = get_tensor_required(gguf, token_embd_name);
    check_tensor(token_embd, {config_->dim, config_->vocab_size}, config_->weight_quantization);
    token_embedding_table_ = tensor_from_gguf<void>(token_embd, device_type);

    std::string output_norm_name = "output_norm.weight";
    const auto& output_norm = get_tensor_required(gguf, output_norm_name);
    check_tensor(output_norm, {config_->dim}, config_->norms_weight_quantization);
    rms_final_weight_ = tensor_from_gguf<void>(output_norm, device_type);

    std::string output_name = "output.weight";
    const auto& output = get_tensor_required(gguf, output_name);
    check_tensor(output, {config_->dim, config_->vocab_size}, config_->weight_quantization);
    wcls_ = tensor_from_gguf<void>(output, device_type);

    // Create blocks
    blocks_.reserve(config_->n_layers);
    for (std::uint32_t i = 0; i < config_->n_layers; ++i) {
        // Create block (following yalm pattern but adapted for GGUF TensorInfo&)
        blocks_.emplace_back(std::make_unique<Block>(gguf, device_type, *config_, i));
    }
}

void Model::forward(InferenceState& s, std::uint32_t token, std::uint32_t pos, InferenceMode mode) {
    if (s.device_type() != device_type_) {
        throw std::runtime_error("InferenceState device type does not match model device type");
    }
    switch (device_type_) {
        case DeviceType::CPU: forward_cpu_(s, token, pos, mode); break;
        case DeviceType::CPU_UnAligned:
        case DeviceType::CUDA:
        default: throw std::runtime_error("Unsupported device type");
    }
}

void Model::forward_cpu_(InferenceState& s, std::uint32_t token, std::uint32_t pos, InferenceMode mode) {
    const auto& c = *config_;

    copy_embedding_(s, token);

    // StreamingLLM
    std::uint32_t kv_sink = pos >= c.max_seq_len ? KV_SINK : 0;
    std::uint32_t kv_pos = kv_sink + ((pos - kv_sink) % (c.max_seq_len - kv_sink));
    std::uint32_t kv_len = pos >= c.max_seq_len ? c.max_seq_len : pos + 1;

    for (auto& b : blocks_) {
        b->block(s, pos, kv_sink, kv_pos, kv_len);
    }

    if (mode == InferenceMode::HydrateKVCache) {
        return;
    }

    // Final layer norm, output to s.xb
    switch (c.norm_type) {
        case LayerNormType::RMSNorm: {
            switch (c.norms_weight_quantization) {
                case QuantizationType::FP32:
                    rmsnorm(s.xb(), s.x(), static_cast<const float*>(rms_final_weight_.data()), c.dim, c.norm_eps);
                    break;
                default: throw std::runtime_error("Unsupported layer norm weight quantization");
            }
            break;
        }
        default: throw std::runtime_error("Unsupported layer norm type");
    }

    // classifier, from s.xb to s.logits
    switch (c.weight_quantization) {
        case QuantizationType::FP16:
            matmul(s.logits(), s.xb(), static_cast<const f16_t*>(wcls_.data()), c.dim, c.vocab_size);
            break;
        default: throw std::runtime_error("Unsupported classifier weight quantization");
    }
}

void Model::copy_embedding_(InferenceState& s, std::uint32_t token) {
    const ModelConfig& c = *config_;
    switch (token_embedding_table_.quantization()) {
        case QuantizationType::FP32: {
            const auto* embtable = static_cast<const float*>(token_embedding_table_.data());
            for (uint32_t i = 0; i < c.dim; ++i) {
                s.x()[i] = embtable[(token * c.dim) + i];
            }
            break;
        }
        case QuantizationType::FP16: {
            const auto* embtable = static_cast<const f16_t*>(token_embedding_table_.data());
            copy_f16_to_f32(&embtable[static_cast<size_t>(token * c.dim)], s.x(), c.dim);
            break;
        }
        default: throw std::runtime_error("Unsupported embedding table quantization type");
    }
}
