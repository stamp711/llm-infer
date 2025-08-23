#pragma once

#include <cstdint>

#include "gguf.hpp"
#include "hf_config.hpp"

enum class QuantizationType : std::uint8_t { FP32, FP16, INT32 };

inline QuantizationType quantization_from_gguf(GGMLType type) {
    switch (type) {
        case GGMLType::GGML_TYPE_F32: return QuantizationType::FP32;
        case GGMLType::GGML_TYPE_F16: return QuantizationType::FP16;
        default: throw std::runtime_error("Unsupported tensor type");
    }
}

inline std::size_t quantization_size(QuantizationType type) {
    switch (type) {
        case QuantizationType::FP32: return 4;
        case QuantizationType::FP16: return 2;
        case QuantizationType::INT32: return 4;
    }
    throw std::runtime_error("Unsupported quantization type");
}

enum class ActivationType : std::uint8_t { GELU, SILU };

enum class LayerNormType : std::uint8_t { RMSNorm };

constexpr std::uint32_t KV_SINK = 4;

struct ModelConfig {
    // Core transformer dimensions
    std::uint32_t n_layers;
    std::uint32_t dim;         // dim of model
    std::uint32_t hidden_dim;  // dim of FFN
    std::uint32_t head_dim;    // dim of each head
    std::uint32_t n_heads;
    std::uint32_t n_kv_heads;  // number of key/value heads, 1 is MQA, n_heads is MHA, between is GQA

    // Vocabulary and sequence
    std::uint32_t vocab_size;
    std::uint32_t max_seq_len;  // context window / kv cache ring buffer size

    // Position encoding
    float rope_theta;          // RoPE theta
    std::uint32_t rotary_dim;  // dimension of rotary position encoding (elements after that don't get rotated)

    // Normalization
    float norm_eps;  // epsilon for layer normalization
    LayerNormType norm_type;

    // Activation and clipping
    ActivationType act;
    bool qkv_clip;
    float qkv_clip_value;  // clip qkv to [-clip, clip]

    // Mixture of experts
    std::uint32_t n_experts;
    std::uint32_t n_experts_active;

    // Quantization settings
    QuantizationType norms_weight_quantization;
    QuantizationType weight_quantization;
    QuantizationType kv_cache_quantization;

    explicit ModelConfig(const HFConfig& hf);
    explicit ModelConfig(const GGUF& gguf);
};
