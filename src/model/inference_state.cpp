#include "model/inference_state.hpp"

#include <cstddef>

#include "model/config.hpp"
#include "model/device.hpp"

InferenceState::InferenceState(const ModelConfig& config, DeviceType device) : config_(&config), device_(device) {
    mode_ = InferenceMode::OutputLogits;

    using QuantizationType::FP32;
    using QuantizationType::INT32;

    x_ = Tensor<float>::allocate(FP32, config_->dim, device_);
    xb_ = Tensor<float>::allocate(FP32, config_->dim, device_);
    xb2_ = Tensor<float>::allocate(FP32, config_->dim, device_);

    hb_ = Tensor<float>::allocate(FP32, config_->hidden_dim, device_);
    hb2_ = Tensor<float>::allocate(FP32, config_->hidden_dim, device_);

    q_ = Tensor<float>::allocate(FP32, static_cast<std::size_t>(config.head_dim) * config.n_heads, device_);
    k_ = Tensor<float>::allocate(FP32, static_cast<std::size_t>(config.head_dim) * config.n_kv_heads, device_);
    v_ = Tensor<float>::allocate(FP32, static_cast<std::size_t>(config.head_dim) * config.n_kv_heads, device_);
    att_ = Tensor<float>::allocate(FP32, static_cast<std::size_t>(config.max_seq_len) * config.n_heads, device_);

    if (config.n_experts > 0) {
        moe_weights_ = Tensor<float>::allocate(FP32, config.n_experts, device_);
        active_experts_weights_ = Tensor<float>::allocate(FP32, config.n_experts_active, device_);
        active_experts_ = Tensor<int>::allocate(INT32, config.n_experts_active, device_);
    }

    logits_ = Tensor<float>::allocate(FP32, config_->vocab_size, DeviceType::CPU);
}
