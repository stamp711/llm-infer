#include "model/block.hpp"

#include <cstddef>

#include "kernels_cpu.hpp"
#include "model/config.hpp"
#include "model/device.hpp"

Block::~Block() {
    if (device_type == DeviceType::CUDA) {
        // TODO: free CUDA resources
    }
}

Block::Block(const GGUF& gguf, DeviceType device_type, const ModelConfig& config, int layer_i)
    : device_type(device_type), config(&config), layer_i(layer_i) {
    // Get all tensors for this layer with shape and quantization checking
    std::string prefix = "blk." + std::to_string(layer_i) + ".";

    // Norm weights
    const auto& attn_norm = get_tensor_required(gguf, prefix + "attn_norm.weight");
    check_tensor(attn_norm, {config.dim}, config.norms_weight_quantization);
    rms_att_weight_ = tensor_from_gguf<void>(attn_norm, device_type);

    const auto& ffn_norm = get_tensor_required(gguf, prefix + "ffn_norm.weight");
    check_tensor(ffn_norm, {config.dim}, config.norms_weight_quantization);
    rms_ffn_weight_ = tensor_from_gguf<void>(ffn_norm, device_type);

    // Attention weights
    const auto& attn_q = get_tensor_required(gguf, prefix + "attn_q.weight");
    check_tensor(attn_q, {config.dim, config.n_heads * config.head_dim}, config.weight_quantization);
    wq = tensor_from_gguf<void>(attn_q, device_type);

    const auto& attn_k = get_tensor_required(gguf, prefix + "attn_k.weight");
    check_tensor(attn_k, {config.dim, config.n_kv_heads * config.head_dim}, config.weight_quantization);
    wk = tensor_from_gguf<void>(attn_k, device_type);

    const auto& attn_v = get_tensor_required(gguf, prefix + "attn_v.weight");
    check_tensor(attn_v, {config.dim, config.n_kv_heads * config.head_dim}, config.weight_quantization);
    wv = tensor_from_gguf<void>(attn_v, device_type);

    const auto& attn_output = get_tensor_required(gguf, prefix + "attn_output.weight");
    check_tensor(attn_output, {config.n_heads * config.head_dim, config.dim}, config.weight_quantization);
    wo = tensor_from_gguf<void>(attn_output, device_type);

    // FFN weights (accounting for MoE)
    bool is_moe = config.n_experts > 0;

    const auto& ffn_gate = get_tensor_required(gguf, prefix + "ffn_gate.weight");
    is_moe ? check_tensor(ffn_gate, {config.dim, config.hidden_dim, config.n_experts}, config.weight_quantization)
           : check_tensor(ffn_gate, {config.dim, config.hidden_dim}, config.weight_quantization);
    w1 = tensor_from_gguf<void>(ffn_gate, device_type);

    const auto& ffn_down = get_tensor_required(gguf, prefix + "ffn_down.weight");
    is_moe ? check_tensor(ffn_down, {config.hidden_dim, config.dim, config.n_experts}, config.weight_quantization)
           : check_tensor(ffn_down, {config.hidden_dim, config.dim}, config.weight_quantization);
    w2 = tensor_from_gguf<void>(ffn_down, device_type);

    const auto& ffn_up = get_tensor_required(gguf, prefix + "ffn_up.weight");
    is_moe ? check_tensor(ffn_up, {config.dim, config.hidden_dim, config.n_experts}, config.weight_quantization)
           : check_tensor(ffn_up, {config.dim, config.hidden_dim}, config.weight_quantization);
    w3 = tensor_from_gguf<void>(ffn_up, device_type);

    // MoE gate (optional)
    if (is_moe) {
        const auto& ffn_gate_inp = get_tensor_required(gguf, prefix + "ffn_gate_inp.weight");
        check_tensor(ffn_gate_inp, {config.dim, config.n_experts}, config.weight_quantization);
        moegate = tensor_from_gguf<void>(ffn_gate_inp, device_type);
    }

    // KV cache
    auto kv_cache_len = config.head_dim * config.n_kv_heads * config.max_seq_len;
    k_cache = Tensor<f16_t>::allocate(QuantizationType::FP16, kv_cache_len, device_type);
    v_cache = Tensor<f16_t>::allocate(QuantizationType::FP16, kv_cache_len, device_type);
}

void Block::block(InferenceState& s, uint32_t pos, uint32_t kv_sink, uint32_t kv_pos, uint32_t kv_len) {
    if (s.device_type() != device_type) {
        throw std::runtime_error("InferenceState device type does not match model device type");
    }
    if (config->norms_weight_quantization != QuantizationType::FP32) {
        throw std::runtime_error("Norms weight quantization type must be FP32");
    }
    switch (device_type) {
        case DeviceType::CPU:
            switch (config->weight_quantization) {
                case QuantizationType::FP16: block_cpu_<f16_t, float>(s, pos, kv_sink, kv_pos, kv_len); return;
                case QuantizationType::FP32:
                // block_cpu_<float, float>(s, pos, kv_sink, kv_pos, kv_len); return;
                default: throw std::runtime_error("Unsupported weight quantization type");
            }

        case DeviceType::CUDA: {
            switch (config->weight_quantization) {
                case QuantizationType::FP16: block_cuda_<f16_t, float>(s, pos, kv_sink, kv_pos, kv_len); return;
                case QuantizationType::FP32:
                default: throw std::runtime_error("Unsupported weight quantization type");
            }
        }

        case DeviceType::CPU_UnAligned:
        default: throw std::runtime_error("Unsupported device type");
    }
}

template <typename WeightT, typename NormT>
void Block::block_cpu_(InferenceState& s, uint32_t pos, uint32_t kv_sink, uint32_t kv_pos, uint32_t kv_len) {
    const ModelConfig& c = *config;

    // ========== Attention ==========

    // Att pre norm, save to s.xb
    switch (c.norm_type) {
        case LayerNormType::RMSNorm: {
            rmsnorm(s.xb(), s.x(), static_cast<const NormT*>(rms_att_weight_.data()), c.dim, c.norm_eps);
            break;
        }
        default: throw std::runtime_error("Unsupported norm type");
    }

    uint32_t q_total_dim = c.n_heads * c.head_dim;      // all heads' q combined dim
    uint32_t kv_total_dim = c.n_kv_heads * c.head_dim;  // all kv-heads' k/v combined dim

    // QKV for current token, save to s.{q,k,v}
    matmul(s.q(), s.xb(), static_cast<const WeightT*>(wq.data()), c.dim, q_total_dim);
    matmul(s.k(), s.xb(), static_cast<const WeightT*>(wk.data()), c.dim, kv_total_dim);
    matmul(s.v(), s.xb(), static_cast<const WeightT*>(wv.data()), c.dim, kv_total_dim);

    // Clip QKV values on s.{q,k,v}
    if (c.qkv_clip) {
        for (uint32_t i = 0; i < q_total_dim; ++i) {
            s.q()[i] = std::min(std::max(s.q()[i], -c.qkv_clip_value), c.qkv_clip_value);
        }
        for (uint32_t i = 0; i < kv_total_dim; ++i) {
            s.k()[i] = std::min(std::max(s.k()[i], -c.qkv_clip_value), c.qkv_clip_value);
            s.v()[i] = std::min(std::max(s.v()[i], -c.qkv_clip_value), c.qkv_clip_value);
        }
    }

    // Apply RoPE on current token's QK on s.{q,k}
    rope(s.q(), c.n_heads, c.head_dim, pos, c.rope_theta, c.rotary_dim);
    rope(s.k(), c.n_kv_heads, c.head_dim, pos, c.rope_theta, c.rotary_dim);

    // Copy k,v into KV cache from s.{q,k}
    for (uint32_t i = 0; i < kv_total_dim; ++i) {
        k_cache.data()[(kv_pos * kv_total_dim) + i] = f32_to_f16(s.k()[i]);
        v_cache.data()[(kv_pos * kv_total_dim) + i] = f32_to_f16(s.v()[i]);
    }

    // Rotate sink tokens' K forward by 1 to maintain relative distance to current token. See StreamingLLM paper.
    // Directly operate on k_cache
    for (uint32_t k = 0; k < kv_sink; ++k) {
        f16_t* sink_k = k_cache.data() + static_cast<ptrdiff_t>(k * kv_total_dim);
        rope(sink_k, c.n_kv_heads, c.head_dim, 1, c.rope_theta, c.rotary_dim);
    }

    // Grouped Query Attention, output to s.attn_out, use s.attn_scores as scores buffer
    uint32_t heads_per_group = c.n_heads / c.n_kv_heads;
#pragma omp parallel for
    for (uint32_t head_i = 0; head_i < c.n_heads; ++head_i) {
        uint32_t group_i = head_i / heads_per_group;
        attn(s.attn_out(head_i), s.attn_scores(head_i), s.q(head_i), k_cache.data(), v_cache.data(), group_i,
             c.head_dim, c.n_kv_heads, kv_len);
    }

    // Transform back to [dim], save to xb
    matmul(s.xb(), s.attn_out(), static_cast<const WeightT*>(wo.data()), q_total_dim, c.dim);

    // residual connection back to x
    for (uint32_t i = 0; i < c.dim; ++i) {
        s.x()[i] += s.xb()[i];
    }

    // ========== FFN ==========

    // FFN pre norm, save to s.xb
    switch (c.norm_type) {
        case LayerNormType::RMSNorm: {
            rmsnorm(s.xb(), s.x(), static_cast<const NormT*>(rms_ffn_weight_.data()), c.dim, c.norm_eps);
            break;
        }
        default: throw std::runtime_error("Unsupported norm type");
    }

    if (c.n_experts > 0) {
        matmul(s.moe_weights(), s.xb(), static_cast<const WeightT*>(moegate.data()), c.dim, c.n_experts);
        moe_gate(s.active_experts_weights(), s.active_experts(), s.moe_weights(), c.n_experts, c.n_experts_active);
    } else {
        s.active_experts()[0] = 0;
        s.active_experts_weights()[0] = 1.F;
    }

    uint32_t n = (c.n_experts > 0) ? c.n_experts_active : 1;
    uint32_t expert_weights_size = c.dim * c.hidden_dim;
    for (uint32_t k = 0; k < n; ++k) {
        uint32_t expert_i = s.active_experts()[k];
        uint32_t expert_weights_offset = expert_i * expert_weights_size;
        float expert_weight = s.active_experts_weights()[k];

        matmul(s.hb(), s.xb(), static_cast<const WeightT*>(w1.data()) + expert_weights_offset, c.dim, c.hidden_dim);
        matmul(s.hb2(), s.xb(), static_cast<const WeightT*>(w3.data()) + expert_weights_offset, c.dim, c.hidden_dim);

        switch (c.act) {
            case ActivationType::GELU:
                for (uint32_t i = 0; i < c.hidden_dim; ++i) {
                    s.hb()[i] = gelu(s.hb()[i]) * s.hb2()[i];
                }
                break;
            case ActivationType::SILU:
                for (uint32_t i = 0; i < c.hidden_dim; ++i) {
                    s.hb()[i] = silu(s.hb()[i]) * s.hb2()[i];
                }
                break;
        }

        // Write to xb2, because xb gets reused in each expert
        matmul(s.xb2(), s.hb(), static_cast<const WeightT*>(w2.data()) + expert_weights_offset, c.hidden_dim, c.dim);

        // residual connection back into x
        for (uint32_t i = 0; i < c.dim; ++i) {
            s.x()[i] += s.xb2()[i] * expert_weight;
        }
    }
}
