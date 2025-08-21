#pragma once

#include <cstddef>

#include "config.hpp"
#include "device.hpp"
#include "model/tensor.hpp"

enum class InferenceMode : std::uint8_t { HydrateKVCache, OutputLogits };

// Always in 32bit float
class InferenceState {
   public:
    explicit InferenceState(const ModelConfig& config, DeviceType device);
    ~InferenceState() = default;

    InferenceState(const InferenceState&) = delete;
    InferenceState(InferenceState&&) = delete;
    InferenceState& operator=(const InferenceState&) = delete;
    InferenceState& operator=(InferenceState&&) = delete;

    [[nodiscard]] float* x() { return x_.get(); }

    [[nodiscard]] float* xb() { return xb_.get(); }
    [[nodiscard]] float* xb(int head) { return xb() + (static_cast<std::ptrdiff_t>(config_->head_dim * head)); }
    [[nodiscard]] float* xb2() { return xb2_.get(); }
    [[nodiscard]] float* xb2(int head) { return xb2() + (static_cast<std::ptrdiff_t>(config_->head_dim * head)); }

    [[nodiscard]] float* hb() { return hb_.get(); }
    [[nodiscard]] float* hb2() { return hb2_.get(); }

    [[nodiscard]] float* q() { return q_.get(); }
    [[nodiscard]] float* k() { return k_.get(); }
    [[nodiscard]] float* v() { return v_.get(); }
    [[nodiscard]] float* att() { return att_.get(); }
    [[nodiscard]] float* att(int head) { return att() + (static_cast<std::ptrdiff_t>(config_->head_dim * head)); }

    [[nodiscard]] float* moe_weights() { return moe_weights_.get(); }
    [[nodiscard]] float* active_experts_weights() { return active_experts_weights_.get(); }
    [[nodiscard]] int* active_experts() { return active_experts_.get(); }

    [[nodiscard]] float* logits() { return logits_.get(); }

    [[nodiscard]] DeviceType device() const { return device_; }
    [[nodiscard]] InferenceMode mode() const { return mode_; }
    void set_mode(InferenceMode mode) { mode_ = mode; }

   private:
    const ModelConfig* config_;
    InferenceMode mode_;
    DeviceType device_;

    // TODO: CUDA-related

    // Activation buffers
    Tensor<float> x_;  // Current activation vector (dim)

    Tensor<float> xb_;   // Residual activation buffer 1 (dim)
    Tensor<float> xb2_;  // Residual activation buffer 2 (dim)

    Tensor<float> hb_;   // Hidden dimension buffer 1 (hidden_dim)
    Tensor<float> hb2_;  // Hidden dimension buffer 2 (hidden_dim)

    // Attention buffers
    Tensor<float> q_;    // Query vectors (n_heads * head_dim)
    Tensor<float> k_;    // Key vectors for current token (n_kv_heads * head_dim)
    Tensor<float> v_;    // Value vectors for current token (n_kv_heads * head_dim)
    Tensor<float> att_;  // Attention scores (n_heads * max_seq_len)

    Tensor<float> moe_weights_;             // Expert weights (n_experts)
    Tensor<float> active_experts_weights_;  // Weights of active experts (n_experts_active)
    Tensor<int> active_experts_;            // Indices of active experts (n_experts_active)

    // Output buffer (always on CPU)
    Tensor<float> logits_;  // Final output logits (vocab_size)
};
