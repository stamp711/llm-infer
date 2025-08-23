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

    [[nodiscard]] float* x() { return x_.data(); }
    [[nodiscard]] float* xb() { return xb_.data(); }
    [[nodiscard]] float* xb2() { return xb2_.data(); }

    [[nodiscard]] float* q() { return q_.data(); }
    [[nodiscard]] float* q(uint32_t head_i) { return q() + (static_cast<std::ptrdiff_t>(config_->head_dim * head_i)); }
    [[nodiscard]] float* k() { return k_.data(); }
    [[nodiscard]] float* v() { return v_.data(); }

    [[nodiscard]] float* attn_scores() { return attn_scores_.data(); }
    [[nodiscard]] float* attn_scores(uint32_t head) {
        return attn_scores() + (static_cast<std::ptrdiff_t>(config_->max_seq_len * head));
    }

    [[nodiscard]] float* attn_out() { return attn_out_.data(); }
    [[nodiscard]] float* attn_out(uint32_t head_i) {
        return attn_out() + (static_cast<std::ptrdiff_t>(config_->head_dim * head_i));
    }

    [[nodiscard]] float* hb() { return hb_.data(); }
    [[nodiscard]] float* hb2() { return hb2_.data(); }

    [[nodiscard]] float* moe_weights() { return moe_weights_.data(); }
    [[nodiscard]] float* active_experts_weights() { return active_experts_weights_.data(); }
    [[nodiscard]] int* active_experts() { return active_experts_.data(); }

    [[nodiscard]] float* logits() { return logits_.data(); }

    [[nodiscard]] DeviceType device_type() const { return device_type_; }
    [[nodiscard]] InferenceMode mode() const { return mode_; }
    void set_mode(InferenceMode mode) { mode_ = mode; }

   private:
    const ModelConfig* config_;
    InferenceMode mode_;
    DeviceType device_type_;

    // TODO: CUDA-related

    // Activation buffers
    Tensor<float> x_;   // Current activation vector (dim)
    Tensor<float> xb_;  // RMSNorm/Attn/FFN results buffer (dim)
    Tensor<float> xb2_;

    // Attention buffers
    Tensor<float> q_;            // Query vectors (n_heads * head_dim)
    Tensor<float> k_;            // Key vectors for current token (n_kv_heads * head_dim)
    Tensor<float> v_;            // Value vectors for current token (n_kv_heads * head_dim)
    Tensor<float> attn_scores_;  // Attention scores (n_heads * max_seq_len)
    Tensor<float> attn_out_;     // Attention output, per-head [head_dim, n_heads]

    Tensor<float> hb_;   // Hidden dimension buffer 1 (hidden_dim)
    Tensor<float> hb2_;  // Hidden dimension buffer 2 (hidden_dim)

    Tensor<float> moe_weights_;             // Expert weights (n_experts)
    Tensor<float> active_experts_weights_;  // Weights of active experts (n_experts_active)
    Tensor<int> active_experts_;            // Indices of active experts (n_experts_active)

    // Output buffer (always on CPU)
    Tensor<float> logits_;  // Final output logits (vocab_size)
};
