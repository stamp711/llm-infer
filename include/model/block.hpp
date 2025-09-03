#pragma once

#include "config.hpp"
#include "device.hpp"
#include "gguf.hpp"
#include "model/inference_state.hpp"
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

    // Compute forward pass for this block and update the inference state accordingly.
    // PRECONDITIONS:
    // - `s.x()` contains the input to the block. Output will also go here.
    // - Block KV cache is hydrated.
    void block(InferenceState& s,  // inference state
               uint32_t pos,       // index of the current token in the sequence
               uint32_t kv_sink,   // number of sink tokens currently in the KV cache
               uint32_t kv_pos,  // index of the current token in the kv cache, must be in [0..kv_len) since kv cache is
                                 // a ring buffer
               uint32_t kv_len   // number of tokens in the kv cache that we will attend over
    );

   private:
    template <typename WeightT, typename NormT>
    void block_cpu_(InferenceState& s, uint32_t pos, uint32_t kv_sink, uint32_t kv_pos, uint32_t kv_len);

    template <typename WeightT, typename NormT>
    void block_cuda_(InferenceState& s, uint32_t pos, uint32_t kv_sink, uint32_t kv_pos, uint32_t kv_len);

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
    Tensor<const void> wo;  // [head_dim * n_heads, dim]

    // weights for ffn
    Tensor<const void> w1;       // [dim, hidden_dim, (n_experts)]
    Tensor<const void> w2;       // [hidden_dim, dim, (n_experts)]
    Tensor<const void> w3;       // [dim, hidden_dim, (n_experts)], for gating
    Tensor<const void> moegate;  // [dim, n_experts], for MoE gating

    // kv cache - these will be allocated and mutable
    Tensor<f16_t> k_cache;  // [head_dim, n_kv_heads, max_seq_len]
    Tensor<f16_t> v_cache;  // [head_dim, n_kv_heads, max_seq_len]
};
