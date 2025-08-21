#pragma once

#include "config.hpp"
#include "device.hpp"
#include "gguf.hpp"
#include "model/tensor.hpp"

using f16_t = std::uint16_t;

// NOTE: weights are stored in column-major order for efficienct computing,
// since they are on the right side of matrix multiplication.
struct Block {
    Block(const Block&) = delete;
    Block(Block&&) = delete;
    Block& operator=(const Block&) = delete;
    Block& operator=(Block&&) = delete;

    Block(const GGUF& gguf, DeviceType device_type, const ModelConfig& config, int layer_i);
    ~Block();

    DeviceType device_type;
    const ModelConfig* config;

    int layer_i;

    // When the tensor has dimensions [x, y, z], the data is laid out in memory such that the innermost
    // (fastest-changing) index corresponds to the first dimension (x), followed by y, then z.

    // weights for norms
    Tensor<const void> rms_att_weight_;  // [dim]
    Tensor<const void> rms_ffn_weight_;  // [dim]

    // weights for attention
    Tensor<const void> wq;  // [dim, head_dim * n_heads]
    Tensor<const void> wk;  // [dim, head_dim * n_kv_heads]
    Tensor<const void> wv;  // [dim, head_dim * n_kv_heads]
    Tensor<const void> wo;  // [dim, head_dim * n_heads]

    // weights for ffn
    Tensor<const void> w1;       // [dim, hidden_dim, (n_experts)]
    Tensor<const void> w2;       // [hidden_dim, dim, (n_experts)]
    Tensor<const void> w3;       // [dim, hidden_dim, (n_experts)], for gating
    Tensor<const void> moegate;  // (dim, n_experts), for MoE gating

    // kv cache - these will be allocated and mutable
    Tensor<f16_t> k_cache;  // [N][n_kv_heads * (head_dim)]
    Tensor<f16_t> v_cache;  // [N][n_kv_heads * (head_dim)]
};
