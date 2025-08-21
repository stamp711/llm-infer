#include <fmt/base.h>
#include <fmt/format.h>
#include <unistd.h>

#include <CLI/CLI.hpp>
#include <filesystem>
#include <magic_enum/magic_enum.hpp>
#include <stdexcept>

#include "gguf.hpp"
#include "model/config.hpp"
#include "model/device.hpp"
#include "model/inference_state.hpp"
#include "model/model.hpp"
#include "tokenizer.hpp"

void print_tensor_info(const std::string& name, const TensorInfo& info) {
    fmt::print("  {:<30} | Type: {:<8} | Shape: [", name, magic_enum::enum_name(info.type));
    for (size_t i = 0; i < info.dimensions.size(); ++i) {
        if (i > 0) fmt::print(", ");
        fmt::print("{}", info.dimensions[i]);
    }
    fmt::println("]");
}

void print_model_tensors(const GGUF& gguf, const std::vector<std::string>& tensor_names) {
    fmt::println("");
    fmt::println("Tensor Information:");
    fmt::println("===================");

    for (const auto& name : tensor_names) {
        const TensorInfo* tensor = gguf.get_tensor(name);
        if (tensor != nullptr) {
            print_tensor_info(name, *tensor);
        } else {
            fmt::println("  {:<30} | NOT FOUND", name);
        }
    }
}

void print_model_config(const ModelConfig& config) {
    fmt::println("");
    fmt::println("Model Configuration from GGUF:");
    fmt::println("==============================");

    fmt::println("Core transformer dimensions:");
    fmt::println("  n_layers: {}", config.n_layers);
    fmt::println("  dim: {}", config.dim);
    fmt::println("  hidden_dim: {}", config.hidden_dim);
    fmt::println("  head_dim: {}", config.head_dim);
    fmt::println("  n_heads: {}", config.n_heads);
    fmt::println("  n_kv_heads: {}", config.n_kv_heads);

    std::string attention_type;
    if (config.n_kv_heads == 1) {
        attention_type = "MQA";
    } else if (config.n_kv_heads == config.n_heads) {
        attention_type = "MHA";
    } else {
        attention_type = "GQA";
    }
    fmt::println("    (Attention type: {})", attention_type);

    fmt::println("");
    fmt::println("Vocabulary and sequence:");
    fmt::println("  vocab_size: {}", config.vocab_size);
    fmt::println("  max_seq_len: {}", config.max_seq_len);

    fmt::println("");
    fmt::println("Position encoding:");
    fmt::println("  rope_theta: {}", config.rope_theta);
    fmt::println("  rotary_dim: {}", config.rotary_dim);

    fmt::println("");
    fmt::println("Normalization:");
    fmt::println("  norm_eps: {}", config.norm_eps);
    fmt::println("  norm_type: {}", magic_enum::enum_name(config.norm_type));

    fmt::println("");
    fmt::println("Activation and clipping:");
    fmt::println("  act: {}", magic_enum::enum_name(config.act));
    fmt::println("  qkv_clip: {}", config.qkv_clip);

    fmt::println("");
    fmt::println("Mixture of experts:");
    fmt::println("  n_experts: {}", config.n_experts);
    fmt::println("  n_experts_active: {}", config.n_experts_active);

    fmt::println("");
    fmt::println("Quantization settings:");
    fmt::println("  norms_weight_quantization: {}", magic_enum::enum_name(config.norms_weight_quantization));
    fmt::println("  weight_quantization: {}", magic_enum::enum_name(config.weight_quantization));
    fmt::println("  kv_cache_quantization: {}", magic_enum::enum_name(config.kv_cache_quantization));

    fmt::println("");
}

int main(int argc, char** argv) {
    CLI::App app{"GGUF Model Testing Tool"};

    std::string gguf_file = "models/Mistral-7B-Instruct-v0.3.fp16.gguf";

    app.add_option("-g,--gguf", gguf_file, "GGUF model file")->check(CLI::ExistingFile);

    CLI11_PARSE(app, argc, argv);

    try {
        fmt::println("Loading GGUF model: {}", gguf_file);

        // Load GGUF file
        GGUF gguf(gguf_file);
        fmt::println("GGUF file loaded successfully");
        fmt::println("Architecture: {}", gguf.architecture());
        fmt::println("Tensor count: {}", gguf.tensor_count());
        fmt::println("Quantization version: {}", gguf.quantization_version());
        fmt::println("Alignment: {}", gguf.alignment());

        // Parse Llama config from GGUF
        auto llama_config = gguf.parse_llama_config();
        if (!llama_config) {
            throw std::runtime_error("Failed to parse Llama config from GGUF file");
        }

        fmt::println("");
        fmt::println("Llama Config parsed successfully");
        fmt::println("Context length: {}", llama_config->context_length);
        fmt::println("Embedding length: {}", llama_config->embedding_length);
        fmt::println("Block count: {}", llama_config->block_count);
        fmt::println("Vocab size: {}", llama_config->vocab_size);

        // Create ModelConfig from GGUF
        ModelConfig model_config(gguf);
        print_model_config(model_config);

        // Test Model loading
        fmt::println("");
        fmt::println("Testing Model loading from GGUF...");
        fmt::println("First few tensor names in GGUF:");
        int count = 0;
        for (const auto& [name, info] : gguf.tensors()) {
            fmt::println("  {}", name);
            if (++count >= 10) break;
        }

        // Print tensor information before model creation
        std::vector<std::string> global_tensors = {"token_embd.weight", "output_norm.weight", "output.weight"};

        // Add layer-specific tensors for first block
        if (model_config.n_layers > 0) {
            global_tensors.insert(global_tensors.end(),
                                  {"blk.0.attn_norm.weight", "blk.0.ffn_norm.weight", "blk.0.attn_q.weight",
                                   "blk.0.attn_k.weight", "blk.0.attn_v.weight", "blk.0.attn_output.weight",
                                   "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight", "blk.0.ffn_up.weight"});
            if (model_config.n_experts > 0) {
                global_tensors.emplace_back("blk.0.ffn_gate_inp.weight");
            }
        }

        print_model_tensors(gguf, global_tensors);

        fmt::println("");
        fmt::println("Creating Model and InferenceState from GGUF...");
        try {
            Model model(gguf, DeviceType::CUDA);
            fmt::println("Model created successfully!");
            fmt::println("  Blocks created: {}", model.blocks.size());
            fmt::println("  Expected blocks: {}", model.config->n_layers);

            auto is = InferenceState(*model.config, model.device_type);
            fmt::println("InferenceState created successfully!");

            CUDAContext::synchronize();

            ::sleep(10);
        } catch (const std::exception& e) {
            fmt::println("Model or InferenceState creation failed: {}", e.what());
            return 1;
        }

        // Load tokenizer from GGUF
        fmt::println("Loading tokenizer from GGUF file");

        Tokenizer tokenizer(gguf);
        fmt::println("Tokenizer loaded successfully");
        fmt::println("Tokenizer vocab size: {}", tokenizer.vocab_size());
        fmt::println("Max accepted token ID: {}", tokenizer.max_token_id());
        fmt::println("BOS token ID: {}", tokenizer.bos_token());
        fmt::println("EOS token ID: {}", tokenizer.eos_token());
        if (tokenizer.eot_token()) {
            fmt::println("EOT token ID: {}", *tokenizer.eot_token());
        }

        // Validate vocabulary size consistency
        if (tokenizer.vocab_size() != static_cast<size_t>(model_config.vocab_size)) {
            fmt::println(stderr, "Warning: Vocab size mismatch - Tokenizer: {}, ModelConfig: {}",
                         tokenizer.vocab_size(), model_config.vocab_size);
            fmt::println(stderr, "Using tokenizer vocab size as source of truth");
        }

        // Test tokenization
        std::string test_text = "Hello, how are you today?";
        fmt::println("");
        fmt::println("Tokenization Test:");
        fmt::println("Text: \"{}\"", test_text);

        auto tokens = tokenizer.encode(test_text);
        fmt::print("Tokens: ");
        for (size_t i = 0; i < tokens.size(); ++i) {
            fmt::print("{}", tokens[i]);
            if (i < tokens.size() - 1) fmt::print(", ");
        }
        fmt::println("");
        fmt::println("Token count: {}", tokens.size());

        fmt::println("");
        fmt::println("Test completed successfully!");

    } catch (const std::exception& e) {
        fmt::println(stderr, "Error: {}", e.what());
        return 1;
    }

    return 0;
}
