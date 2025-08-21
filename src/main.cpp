#include <CLI/CLI.hpp>
#include <algorithm>
#include <cfloat>
#include <iomanip>
#include <iostream>
#include <magic_enum/magic_enum.hpp>
#include "safetensors.hpp"
#include <string>
#include <tensor_types.hpp>
#include <unordered_map>
#include <fstream>
#include <filesystem>
#include "hf_config.hpp"
#include "tokenizer.hpp"
#include "model/config.hpp"

void print_model_config(const ModelConfig& config) {
    std::cout << "\nModel Configuration:\n";
    std::cout << "====================\n";
    
    // Model architecture
    std::cout << "Architecture:\n";
    std::cout << "  Layers: " << config.n_layers << "\n";
    std::cout << "  Dimensions: " << config.dim << "\n";
    std::cout << "  Hidden dimension: " << config.hidden_dim << "\n";
    std::cout << "  Head dimension: " << config.head_dim << "\n";
    
    // Attention configuration
    std::cout << "\nAttention:\n";
    std::cout << "  Query heads: " << config.n_heads << "\n";
    std::cout << "  Key/Value heads: " << config.n_kv_heads << "\n";
    
    std::string attention_type;
    if (config.n_kv_heads == 1) {
        attention_type = "Multi-Query Attention (MQA)";
    } else if (config.n_kv_heads == config.n_heads) {
        attention_type = "Multi-Head Attention (MHA)";
    } else {
        attention_type = "Grouped-Query Attention (GQA)";
    }
    std::cout << "  Attention type: " << attention_type << "\n";
    
    // Vocabulary and sequence
    std::cout << "\nVocabulary & Sequence:\n";
    std::cout << "  Vocab size: " << config.vocab_size << "\n";
    std::cout << "  Max sequence length: " << config.max_seq_len << "\n";
    
    // Position encoding
    std::cout << "\nPosition Encoding:\n";
    std::cout << "  RoPE theta: " << config.rope_theta << "\n";
    std::cout << "  Rotary dimensions: " << config.rotary_dim << "/" << config.head_dim;
    if (config.rotary_dim < config.head_dim) {
        float partial_factor = static_cast<float>(config.rotary_dim) / static_cast<float>(config.head_dim);
        std::cout << " (" << std::fixed << std::setprecision(2) << partial_factor << " partial rotation)";
    }
    std::cout << "\n";
    
    // Normalization
    std::cout << "\nNormalization:\n";
    std::cout << "  Type: ";
    switch (config.norm_type) {
        case LayerNormType::RMSNorm:
            std::cout << "RMS Normalization";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << "\n";
    std::cout << "  Epsilon: " << config.norm_eps << "\n";
    
    // Activation function
    std::cout << "\nActivation:\n";
    std::cout << "  Function: ";
    switch (config.act) {
        case ActivationType::GELU:
            std::cout << "GELU";
            break;
        case ActivationType::SILU:
            std::cout << "SiLU (Swish)";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << "\n";
    
    // QKV clipping
    if (config.qkv_clip < FLT_MAX) {
        std::cout << "  QKV clipping: ±" << config.qkv_clip << "\n";
    } else {
        std::cout << "  QKV clipping: None\n";
    }
    
    // Mixture of Experts
    if (config.n_experts > 0) {
        std::cout << "\nMixture of Experts:\n";
        std::cout << "  Total experts: " << config.n_experts << "\n";
        std::cout << "  Active experts: " << config.n_experts_active << "\n";
        std::cout << "  Expert utilization: " << std::fixed << std::setprecision(1) 
                  << (100.0f * config.n_experts_active / config.n_experts) << "%\n";
    }
    
    // Quantization
    std::cout << "\nQuantization:\n";
    std::cout << "  Weights: ";
    switch (config.weight_quantization) {
        case QuantizationType::FP32:
            std::cout << "FP32 (32-bit float)";
            break;
        case QuantizationType::FP16:
            std::cout << "FP16 (16-bit float)";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << "\n";
    
    std::cout << "  Norms: ";
    switch (config.norms_weight_quantization) {
        case QuantizationType::FP32:
            std::cout << "FP32 (32-bit float)";
            break;
        case QuantizationType::FP16:
            std::cout << "FP16 (16-bit float)";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << "\n";
    
    std::cout << "  KV Cache: ";
    switch (config.kv_cache_quantization) {
        case QuantizationType::FP32:
            std::cout << "FP32 (32-bit float)";
            break;
        case QuantizationType::FP16:
            std::cout << "FP16 (16-bit float)";
            break;
        default:
            std::cout << "Unknown";
    }
    std::cout << "\n";
    
    // Memory estimates
    std::cout << "\nMemory Estimates:\n";
    size_t params_per_layer = static_cast<size_t>(config.dim) * config.dim * 4 + // Self-attention weights
                              static_cast<size_t>(config.dim) * config.hidden_dim * 3; // FFN weights
    size_t total_params = static_cast<size_t>(config.vocab_size) * config.dim + // Embeddings
                          static_cast<size_t>(config.n_layers) * params_per_layer + // Layers
                          config.dim; // Final norm
    
    std::cout << "  Estimated parameters: " << total_params / 1000000 << "M\n";
    
    size_t bytes_per_param = (config.weight_quantization == QuantizationType::FP16) ? 2 : 4;
    size_t model_size_bytes = total_params * bytes_per_param;
    
    std::cout << "  Model size: ";
    if (model_size_bytes >= 1024 * 1024 * 1024) {
        std::cout << std::fixed << std::setprecision(2) << model_size_bytes / (1024.0 * 1024.0 * 1024.0) << " GB";
    } else {
        std::cout << std::fixed << std::setprecision(2) << model_size_bytes / (1024.0 * 1024.0) << " MB";
    }
    std::cout << "\n";
}

// Create read-only dynamic tensor view from SafeTensors
template <typename T>
[[nodiscard]] tensor::TensorDynamicView<T> from_safetensor(const safetensors::TensorView& view) {
    if constexpr (std::is_same_v<T, float>) {
        if (view.dtype() != safetensors::Dtype::F32) {
            throw std::runtime_error("Dtype mismatch: expected F32");
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (view.dtype() != safetensors::Dtype::F64) {
            throw std::runtime_error("Dtype mismatch: expected F64");
        }
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    const auto* data_ptr = reinterpret_cast<const T*>(view.data().data());
    const auto& shape = view.shape();
    std::size_t total_size = 1;
    for (auto dim : shape) total_size *= dim;
    return tensor::TensorDynamicView<T>(shape, tensor::ViewStorage<T>(data_ptr, total_size));
}

void print_safetensors_info(const safetensors::SafeTensors& st) {
    std::cout << "\nSafeTensors Information:\n";
    std::cout << "========================\n\n";

    std::cout << "Total tensors: " << st.size() << "\n";
    if (st.empty()) {
        std::cout << "No tensors found.\n";
        return;
    }

    // Calculate total file statistics
    std::size_t total_parameters = 0;
    std::size_t total_data_size = 0;
    std::unordered_map<std::string, std::size_t> dtype_counts;
    std::unordered_map<std::string, std::size_t> dtype_sizes;

    auto tensor_names = st.names();
    // Sort tensor names with proper numeric ordering by splitting on '.'
    std::ranges::sort(tensor_names, [](const std::string_view& a, const std::string_view& b) {
        auto split = [](std::string_view str) {
            std::vector<std::string> parts;
            size_t start = 0;
            size_t pos = 0;
            while ((pos = str.find('.', start)) != std::string_view::npos) {
                parts.emplace_back(str.substr(start, pos - start));
                start = pos + 1;
            }
            parts.emplace_back(str.substr(start));
            return parts;
        };

        auto parts_a = split(a);
        auto parts_b = split(b);

        for (size_t i = 0; i < std::min(parts_a.size(), parts_b.size()); ++i) {
            // Try to parse as numbers first
            bool is_num_a = !parts_a[i].empty() && std::all_of(parts_a[i].begin(), parts_a[i].end(), ::isdigit);
            bool is_num_b = !parts_b[i].empty() && std::all_of(parts_b[i].begin(), parts_b[i].end(), ::isdigit);

            if (is_num_a && is_num_b) {
                int num_a = std::stoi(parts_a[i]);
                int num_b = std::stoi(parts_b[i]);
                if (num_a != num_b) return num_a < num_b;
            } else {
                if (parts_a[i] != parts_b[i]) return parts_a[i] < parts_b[i];
            }
        }
        return parts_a.size() < parts_b.size();
    });

    for (size_t i = 0; i < tensor_names.size(); ++i) {
        const auto& name = tensor_names[i];
        auto tensor_view = st.tensor(name);

        std::cout << "Tensor #" << i + 1 << ":\n";
        std::cout << "  Name: " << name << "\n";

        auto dtype_name = std::string(magic_enum::enum_name(tensor_view.dtype()));
        std::cout << "  Dtype: " << dtype_name << "\n";

        std::cout << "  Shape: [";
        const auto& shape = tensor_view.shape();
        for (size_t j = 0; j < shape.size(); ++j) {
            std::cout << shape[j];
            if (j < shape.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";

        std::cout << "  Dimensions: " << shape.size() << "D\n";

        std::size_t total_elements = 1;
        for (auto dim : shape) {
            total_elements *= dim;
        }
        std::cout << "  Elements: " << total_elements << "\n";

        auto data_size = tensor_view.data_len();
        std::cout << "  Data size: " << data_size << " bytes";
        if (data_size >= 1024 * 1024) {
            std::cout << " (" << std::fixed << std::setprecision(2) << data_size / (1024.0 * 1024.0) << " MB)";
        } else if (data_size >= 1024) {
            std::cout << " (" << std::fixed << std::setprecision(2) << data_size / 1024.0 << " KB)";
        }
        std::cout << "\n";

        auto element_size = safetensors::dtype_bitsize(tensor_view.dtype()) / 8;
        std::cout << "  Element size: " << element_size << " bytes\n";

        // Accumulate statistics
        total_parameters += total_elements;
        total_data_size += data_size;
        dtype_counts[dtype_name]++;
        dtype_sizes[dtype_name] += data_size;

        std::cout << "\n";
    }

    // Print summary statistics
    std::cout << "File Summary:\n";
    std::cout << "=============\n";
    std::cout << "Total parameters: " << total_parameters << "\n";
    std::cout << "Total data size: " << total_data_size << " bytes";
    if (total_data_size >= 1024 * 1024 * 1024) {
        std::cout << " (" << std::fixed << std::setprecision(2) << total_data_size / (1024.0 * 1024.0 * 1024.0)
                  << " GB)";
    } else if (total_data_size >= 1024 * 1024) {
        std::cout << " (" << std::fixed << std::setprecision(2) << total_data_size / (1024.0 * 1024.0) << " MB)";
    }
    std::cout << "\n";

    std::cout << "\nData type breakdown:\n";
    for (const auto& [dtype, count] : dtype_counts) {
        std::cout << "  " << dtype << ": " << count << " tensor(s), ";
        auto size = dtype_sizes[dtype];
        std::cout << size << " bytes";
        if (size >= 1024 * 1024) {
            std::cout << " (" << std::fixed << std::setprecision(2) << size / (1024.0 * 1024.0) << " MB)";
        }
        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    CLI::App app{"LLM Inference Tool"};

    std::string config_file;
    std::string tokenizer_file;
    std::string safetensors_file;
    bool show_tensors = false;

    app.add_option("-c,--config", config_file, "HuggingFace config.json file")->required();
    app.add_option("-t,--tokenizer", tokenizer_file, "HuggingFace tokenizer.json file")->required();
    app.add_option("-s,--safetensors", safetensors_file, "SafeTensors weights file")->required();
    app.add_flag("--show-tensors", show_tensors, "Show detailed tensor information");

    CLI11_PARSE(app, argc, argv);

    try {
        // Verify all files exist
        if (!std::filesystem::exists(config_file)) {
            throw std::runtime_error("Config file not found: " + config_file);
        }
        if (!std::filesystem::exists(tokenizer_file)) {
            throw std::runtime_error("Tokenizer file not found: " + tokenizer_file);
        }
        if (!std::filesystem::exists(safetensors_file)) {
            throw std::runtime_error("SafeTensors file not found: " + safetensors_file);
        }

        std::cout << "Loading model files:\n";
        std::cout << "  Config: " << config_file << "\n";
        std::cout << "  Tokenizer: " << tokenizer_file << "\n";
        std::cout << "  Weights: " << safetensors_file << "\n\n";
        
        safetensors::SafeTensors st(safetensors_file);
        std::cout << "SafeTensors file loaded successfully\n";
        
        // Load and print model configuration
        try {
            HFConfig hf_config(config_file);
            ModelConfig model_config(hf_config);
            
            print_model_config(model_config);
            
            // Load and test tokenizer
            const auto& config = hf_config.config();
            try {
                std::cout << "\nTokenizer Test:\n";
                std::cout << "===============\n";
                Tokenizer tokenizer(tokenizer_file, 
                                  config["bos_token_id"].get<std::uint32_t>(), 
                                  config["eos_token_id"].get<std::uint32_t>(),
                                  config["vocab_size"].get<std::size_t>());
                std::cout << "Tokenizer vocab size: " << tokenizer.vocab_size() << "\n";
                
                // Quick tokenization test
                std::string test_text = "Hello, world!";
                auto tokens = tokenizer.encode(test_text);
                std::cout << "Test: \"" << test_text << "\" -> ";
                for (size_t i = 0; i < std::min(size_t(5), tokens.size()); ++i) {
                    std::cout << tokens[i] << " ";
                }
                if (tokens.size() > 5) std::cout << "...";
                std::cout << "\n";
                
            } catch (const std::exception& e) {
                std::cout << "Tokenizer error: " << e.what() << "\n";
            }
            
        } catch (const std::exception& e) {
            std::cout << "Model config error: " << e.what() << "\n";
        }
        
        // Print SafeTensors metadata if present
        const auto& metadata = st.metadata();
        if (metadata.metadata.has_value() && !metadata.metadata->empty()) {
            std::cout << "\nSafeTensors Metadata:\n";
            std::cout << "=====================\n";
            for (const auto& [key, value] : *metadata.metadata) {
                std::cout << key << ": " << value << "\n";
            }
        }
        
        std::cout << "\nTotal tensors: " << st.size() << "\n";

        if (!st.empty()) {
            // Show basic summary even without -t flag
            std::size_t total_parameters = 0;
            std::size_t total_data_size = 0;

            auto tensor_names = st.names();
            for (const auto& name : tensor_names) {
                auto tensor_view = st.tensor(name);
                std::size_t elements = 1;
                for (auto dim : tensor_view.shape()) {
                    elements *= dim;
                }
                total_parameters += elements;
                total_data_size += tensor_view.data_len();
            }

            std::cout << "Total parameters: " << total_parameters << "\n";
            std::cout << "Total data size: " << total_data_size << " bytes";
            if (total_data_size >= 1024 * 1024 * 1024) {
                std::cout << " (" << std::fixed << std::setprecision(2) << total_data_size / (1024.0 * 1024.0 * 1024.0)
                          << " GB)";
            } else if (total_data_size >= 1024 * 1024) {
                std::cout << " (" << std::fixed << std::setprecision(2) << total_data_size / (1024.0 * 1024.0)
                          << " MB)";
            }
            std::cout << "\n";
        }

        if (show_tensors) {
            print_safetensors_info(st);
        } else if (!st.empty()) {
            std::cout << "\nUse -t flag to see detailed tensor information.\n";
        }

        // Demonstrate tensor usage with first float tensor
        auto names = st.names();
        for (const auto& name : names) {
            auto view = st.tensor(name);
            if (view.dtype() == safetensors::Dtype::F32) {
                std::cout << "\nTensor demonstration with: " << name << "\n";
                auto tensor = from_safetensor<float>(view);
                std::cout << "Created tensor view with " << tensor.size() << " elements\n";
                std::cout << "First few values: ";
                for (size_t i = 0; i < std::min(size_t(5), tensor.size()); ++i) {
                    std::cout << tensor[i] << " ";
                }
                std::cout << "\n";
                break;
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
