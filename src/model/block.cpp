#include "model/block.hpp"

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
