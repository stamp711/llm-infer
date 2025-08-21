#include "model/config.hpp"

#include <cfloat>
#include <iostream>
#include <stdexcept>

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

    qkv_clip = FLT_MAX;  // no clipping by default

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

    kv_cache_quantization = QuantizationType::FP16;

    // Validation
    if (dim % n_heads != 0) {
        throw std::runtime_error("hidden_size must be divisible by num_attention_heads");
    }
}
