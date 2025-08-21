#pragma once

#include <cstdint>

#include "hf_config.hpp"

enum class DeviceType : std::uint8_t { CPU, CUDA };

enum class QuantizationType : std::uint8_t { FP32, FP16 };

enum class ActivationType : std::uint8_t { GELU, SILU };

enum class LayerNormType : std::uint8_t { RMSNorm };

struct ModelConfig {
    // Core transformer dimensions
    int n_layers;
    int dim;         // dim of model
    int hidden_dim;  // dim of FFN
    int head_dim;    // dim of each head
    int n_heads;
    int n_kv_heads;  // number of key/value heads, 1 is MQA, n_heads is MHA, between is GQA

    // Vocabulary and sequence
    int vocab_size;
    int max_seq_len;  // context window / kv cache ring buffer size

    // Position encoding
    float rope_theta;  // RoPE theta
    int rotary_dim;    // dimension of rotary position encoding (elements after that don't get rotated)

    // Normalization
    float norm_eps;  // epsilon for layer normalization
    LayerNormType norm_type;

    // Activation and clipping
    ActivationType act;
    float qkv_clip;  // clip qkv to [-clip, clip]

    // Mixture of experts
    int n_experts;
    int n_experts_active;

    // Quantization settings
    QuantizationType norms_weight_quantization;
    QuantizationType weight_quantization;
    QuantizationType kv_cache_quantization;

    explicit ModelConfig(const HFConfig& hf);
};
