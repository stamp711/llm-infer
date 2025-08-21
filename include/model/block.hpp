#pragma once

#include "config.hpp"

// NOTE: weights are stored in column-major order for efficienct computing,
// since they are on the right side of matrix multiplication.
struct Block {
    Block();

    int layer_i;
    ModelConfig* model_config;
    DeviceType device_type;

    // weights for norms
    void* rms_att_weight_ = nullptr;  // (dim)
    void* rms_ffn_weight_ = nullptr;  // (dim)

    // weights for attention
    void* wq = nullptr;  // n_heads * (dim, head_dim)
    void* wk = nullptr;  // n_kv_heads * (dim, head_dim)
    void* wv = nullptr;  // (n_kv_heads * head_dim, dim)
    void* wo = nullptr;  // (n_heads * head_dim, dim)

    // weights for ffn
    void* w1 = nullptr;       // n_experts? * (dim, hidden_dim)
    void* w2 = nullptr;       // n_experts? * (hidden_dim, dim)
    void* w3 = nullptr;       // n_experts? * (dim, hidden_dim), for gating
    void* moegate = nullptr;  // (dim, n_experts), for MoE gating

    // kv cache
    void* k_cache = nullptr;  // [N][n_kv_heads * (head_dim)]
    void* v_cache = nullptr;  // [N][n_kv_heads * (head_dim)]
};
