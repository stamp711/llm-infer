#include "model/config.hpp"

#include <cfloat>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <unordered_set>

#include "gguf.hpp"

QuantizationType quantization_from_gguf(const GGMLType& type) {
    switch (type) {
        case GGMLType::GGML_TYPE_F32: return QuantizationType::FP32;
        case GGMLType::GGML_TYPE_F16: return QuantizationType::FP16;
        default: throw std::runtime_error("Unsupported tensor type");
    }
}

namespace {

struct QuantizationInfo {
    QuantizationType weight_quantization;
    QuantizationType norms_weight_quantization;
};

QuantizationType ggml_type_to_quantization_type(GGMLType type) {
    switch (type) {
        case GGMLType::GGML_TYPE_F32: return QuantizationType::FP32;
        case GGMLType::GGML_TYPE_F16: return QuantizationType::FP16;
        default:
            throw std::runtime_error("Unsupported GGML type for quantization: " +
                                     std::to_string(static_cast<uint32_t>(type)));
    }
}

QuantizationInfo detect_quantization_from_tensors(const GGUF& gguf) {
    // Find norm tensors
    const static auto norm_tensor_names = {"output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ffn_norm.weight"};
    std::optional<QuantizationType> norm_quant;
    for (const auto& norm_name : norm_tensor_names) {
        if (const auto* tensor = gguf.get_tensor(norm_name)) {
            norm_quant = ggml_type_to_quantization_type(tensor->type);
            break;
        }
    }
    if (!norm_quant.has_value()) {
        throw std::runtime_error("Could not find any norm tensors to determine quantization.");
    }

    // Find weight tensors
    const static auto weight_tensor_names = {"blk.0.attn_q.weight", "blk.0.attn_k.weight", "blk.0.attn_v.weight",
                                             "blk.0.ffn_up.weight"};
    std::unordered_set<QuantizationType> weight_quant_types;
    for (const auto& weight_name : weight_tensor_names) {
        if (const auto* tensor = gguf.get_tensor(weight_name)) {
            weight_quant_types.insert(ggml_type_to_quantization_type(tensor->type));
        }
    }
    if (weight_quant_types.empty()) {
        throw std::runtime_error(
            "Could not find any weight tensors to determine quantization. Expected one of: blk.0.attn_q.weight, "
            "blk.0.attn_k.weight, blk.0.attn_v.weight, blk.0.ffn_up.weight");
    }
    if (weight_quant_types.size() > 1) {
        throw std::runtime_error(
            "Inconsistent quantization types found in weight tensors. Mixed quantization is not supported.");
    }

    return {.weight_quantization = *weight_quant_types.begin(), .norms_weight_quantization = *norm_quant};
}

}  // namespace

ModelConfig::ModelConfig(const HFConfig& hf) {
    const auto& config = hf.config();

    n_layers = config.at("num_hidden_layers").get<int>();
    dim = config.at("hidden_size").get<int>();
    hidden_dim = config.at("intermediate_size").get<int>();
    n_heads = config.at("num_attention_heads").get<int>();
    n_kv_heads = config.value("num_key_value_heads", n_heads);
    head_dim = config.value("head_dim", dim / n_heads);

    vocab_size = config.at("vocab_size").get<int>();
    max_seq_len = config.at("max_position_embeddings").get<int>();

    // Position encoding (rope_theta optional, rotary_dim computed)
    rope_theta = config.value("rope_theta", 10000.0F);
    float partial_rotary_factor = config.value("partial_rotary_factor", 1.0F);
    rotary_dim = static_cast<int>(static_cast<float>(head_dim) * partial_rotary_factor);

    // Normalization
    norm_eps = config.value("rms_norm_eps", 1e-5F);
    norm_type = LayerNormType::RMSNorm;

    // Activation and clipping
    std::string act_str = config.value("hidden_act", std::string("gelu"));
    if (act_str == "silu") {
        act = ActivationType::SILU;
    } else if (act_str == "gelu") {
        act = ActivationType::GELU;
    } else {
        std::cerr << "Unsupported activation function: " << act_str << ", defaulting to GELU\n";
        act = ActivationType::GELU;
    }

    qkv_clip = false;
    qkv_clip_value = FLT_MAX;  // no clipping by default

    // Mixture of experts
    if (config.contains("num_local_experts")) {
        n_experts = config.at("num_local_experts").get<int>();
        n_experts_active = config.at("num_experts_per_tok").get<int>();
    } else {
        n_experts = 0;  // standard model
        n_experts_active = 0;
    }

    // Quantization settings based on model dtype
    std::string dtype_str = config.at("torch_dtype").get<std::string>();
    if (dtype_str == "float32") {
        weight_quantization = QuantizationType::FP32;
        norms_weight_quantization = QuantizationType::FP32;
    } else if (dtype_str == "float16") {
        weight_quantization = QuantizationType::FP16;
        norms_weight_quantization = QuantizationType::FP16;
    } else {
        throw std::runtime_error("Unsupported torch_dtype: " + dtype_str);
    }

    kv_cache_quantization = QuantizationType::FP16;  // Fixed for now

    // Validation
    if (dim % n_heads != 0) {
        throw std::runtime_error("hidden_size must be divisible by num_attention_heads");
    }
}

ModelConfig::ModelConfig(const GGUF& gguf) {
    // Parse Llama config from GGUF metadata
    auto llama_config = gguf.parse_llama_config();
    if (!llama_config.has_value()) {
        throw std::runtime_error("Failed to parse Llama configuration from GGUF. Architecture must be 'llama'.");
    }

    const auto& llama = *llama_config;
    n_layers = static_cast<std::uint32_t>(llama.block_count);
    dim = static_cast<std::uint32_t>(llama.embedding_length);
    hidden_dim = static_cast<std::uint32_t>(llama.feed_forward_length);
    n_heads = static_cast<std::uint32_t>(llama.attention_head_count);
    n_kv_heads = static_cast<std::uint32_t>(llama.kv_head_count());
    head_dim = static_cast<std::uint32_t>(llama.head_dim());

    vocab_size = static_cast<std::uint32_t>(llama.vocab_size);
    max_seq_len = static_cast<std::uint32_t>(llama.context_length);

    rope_theta = llama.rope_freq_base;
    rotary_dim = static_cast<std::uint32_t>(llama.rope_dimension_count);

    norm_eps = llama.layer_norm_rms_epsilon;
    norm_type = LayerNormType::RMSNorm;  // GGUF only supports RMS normalization

    act = ActivationType::SILU;  // Default for Llama models, not stored in GGUF metadata
    qkv_clip = false;
    qkv_clip_value = FLT_MAX;  // No clipping by default, not stored in GGUF metadata

    n_experts = static_cast<std::uint32_t>(llama.expert_count.value_or(0));
    n_experts_active = static_cast<std::uint32_t>(llama.expert_used_count.value_or(0));

    // Detect quantization from actual tensor types
    auto quant_info = detect_quantization_from_tensors(gguf);
    weight_quantization = quant_info.weight_quantization;
    norms_weight_quantization = quant_info.norms_weight_quantization;
    kv_cache_quantization = QuantizationType::FP16;  // Runtime choice

    // Comprehensive validation with specific error messages
    if (dim % n_heads != 0) {
        throw std::runtime_error("embedding_length (" + std::to_string(dim) +
                                 ") must be divisible by attention_head_count (" + std::to_string(n_heads) + ")");
    }

    if (n_experts > 0 && n_experts_active == 0) {
        throw std::runtime_error("If n_experts > 0 (" + std::to_string(n_experts) +
                                 "), then n_experts_active must also be > 0");
    }

    if (n_experts_active > n_experts) {
        throw std::runtime_error("n_experts_active (" + std::to_string(n_experts_active) +
                                 ") cannot be greater than n_experts (" + std::to_string(n_experts) + ")");
    }

    // Validate reasonable ranges - uint32_t is always >= 0, so just check for 0
    if (n_layers == 0 || dim == 0 || hidden_dim == 0 || n_heads == 0 || n_kv_heads == 0) {
        throw std::runtime_error("All model dimensions must be positive. Got: n_layers=" + std::to_string(n_layers) +
                                 ", dim=" + std::to_string(dim) + ", hidden_dim=" + std::to_string(hidden_dim) +
                                 ", n_heads=" + std::to_string(n_heads) + ", n_kv_heads=" + std::to_string(n_kv_heads));
    }
}
