#include "model/inference_state.hpp"

#include <cstddef>

#include "model/config.hpp"
#include "model/device.hpp"

InferenceState::InferenceState(const ModelConfig& config, DeviceType device) : config_(&config), device_type_(device) {
    mode_ = InferenceMode::OutputLogits;

    using QuantizationType::FP32;
    using QuantizationType::INT32;

    x_ = Tensor<float>::allocate(FP32, config_->dim, device_type_);
    xb_ = Tensor<float>::allocate(FP32, config_->dim, device_type_);
    xb2_ = Tensor<float>::allocate(FP32, config_->dim, device_type_);

    q_ = Tensor<float>::allocate(FP32, static_cast<std::size_t>(config.head_dim) * config.n_heads, device_type_);
    k_ = Tensor<float>::allocate(FP32, static_cast<std::size_t>(config.head_dim) * config.n_kv_heads, device_type_);
    v_ = Tensor<float>::allocate(FP32, static_cast<std::size_t>(config.head_dim) * config.n_kv_heads, device_type_);
    attn_scores_ =
        Tensor<float>::allocate(FP32, static_cast<std::size_t>(config.max_seq_len) * config.n_heads, device_type_);
    attn_out_ =
        Tensor<float>::allocate(FP32, static_cast<std::size_t>(config_->head_dim) * config_->n_heads, device_type_);

    hb_ = Tensor<float>::allocate(FP32, config_->hidden_dim, device_type_);
    hb2_ = Tensor<float>::allocate(FP32, config_->hidden_dim, device_type_);

    bool is_moe = config.n_experts > 0;
    moe_weights_ = Tensor<float>::allocate(FP32, is_moe ? config.n_experts : 1, device_type_);
    active_experts_weights_ = Tensor<float>::allocate(FP32, is_moe ? config.n_experts_active : 1, device_type_);
    active_experts_ = Tensor<int>::allocate(INT32, is_moe ? config.n_experts_active : 1, device_type_);

    logits_ = Tensor<float>::allocate(FP32, config_->vocab_size, DeviceType::CPU);
}
