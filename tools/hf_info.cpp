#include <fmt/format.h>

#include <CLI/CLI.hpp>
#include <filesystem>

#include "hf_config.hpp"
#include "safetensors.hpp"

void print_config_info(const std::string& config_path) {
    try {
        HFConfig config(config_path);
        const auto& json_config = config.config();

        fmt::println("=== Configuration ({}) ===", config_path);

        if (json_config.contains("model_type")) {
            fmt::println("Model Type: {}", json_config["model_type"].get<std::string>());
        }

        if (json_config.contains("architectures") && json_config["architectures"].is_array()) {
            fmt::print("Architectures: ");
            for (const auto& arch : json_config["architectures"]) {
                fmt::print("{} ", arch.get<std::string>());
            }
            fmt::println("");
        }

        if (json_config.contains("vocab_size")) {
            fmt::println("Vocab Size: {}", json_config["vocab_size"].get<int>());
        }

        if (json_config.contains("hidden_size")) {
            fmt::println("Hidden Size: {}", json_config["hidden_size"].get<int>());
        }

        if (json_config.contains("num_hidden_layers")) {
            fmt::println("Hidden Layers: {}", json_config["num_hidden_layers"].get<int>());
        }

        if (json_config.contains("num_attention_heads")) {
            fmt::println("Attention Heads: {}", json_config["num_attention_heads"].get<int>());
        }

        if (json_config.contains("max_position_embeddings")) {
            fmt::println("Max Position Embeddings: {}", json_config["max_position_embeddings"].get<int>());
        }

        fmt::println("\nFull Configuration:");
        fmt::println("{}", json_config.dump(2));

    } catch (const std::exception& e) {
        fmt::println(stderr, "Error reading config: {}", e.what());
    }
}

void print_safetensors_info(const std::string& safetensors_path) {
    try {
        safetensors::SafeTensors tensors(safetensors_path);

        fmt::println("\n=== SafeTensors ({}) ===", safetensors_path);
        fmt::println("Number of tensors: {}", tensors.size());

        const auto& metadata = tensors.metadata();
        if (metadata.metadata) {
            fmt::println("\nMetadata:");
            for (const auto& [key, value] : *metadata.metadata) {
                fmt::println("  {}: {}", key, value);
            }
        }

        fmt::println("\nTensor Information:");
        std::size_t total_params = 0;

        for (const auto& tensor_name : tensors.names()) {
            auto tensor_view = tensors.tensor(tensor_name);
            const auto& shape = tensor_view.shape();

            fmt::print("  {}: ", tensor_name);

            // Print dtype
            std::string dtype_str;
            switch (tensor_view.dtype()) {
                case safetensors::Dtype::F32: dtype_str = "F32"; break;
                case safetensors::Dtype::F16: dtype_str = "F16"; break;
                case safetensors::Dtype::BF16: dtype_str = "BF16"; break;
                case safetensors::Dtype::I32: dtype_str = "I32"; break;
                case safetensors::Dtype::I16: dtype_str = "I16"; break;
                case safetensors::Dtype::I8: dtype_str = "I8"; break;
                case safetensors::Dtype::U8: dtype_str = "U8"; break;
                case safetensors::Dtype::BOOL: dtype_str = "BOOL"; break;
                default: dtype_str = "UNKNOWN"; break;
            }
            fmt::print("{} ", dtype_str);

            // Print shape
            fmt::print("[");
            for (std::size_t i = 0; i < shape.size(); ++i) {
                if (i > 0) fmt::print(", ");
                fmt::print("{}", shape[i]);
            }
            fmt::print("]");

            // Calculate number of parameters
            std::size_t tensor_params = 1;
            for (auto dim : shape) {
                tensor_params *= dim;
            }
            total_params += tensor_params;

            fmt::println(" ({} parameters, {} bytes)", tensor_params, tensor_view.data_len());
        }

        fmt::println("\nTotal Parameters: {}", total_params);
        fmt::println("Total Size: {:.2f} MB",
                     static_cast<double>(std::filesystem::file_size(safetensors_path)) / (1024.0 * 1024.0));

    } catch (const std::exception& e) {
        fmt::println(stderr, "Error reading safetensors: {}", e.what());
    }
}

int main(int argc, char* argv[]) {
    CLI::App app{"HuggingFace Model Information Tool"};

    std::string config_path;
    std::string safetensors_path;

    app.add_option("-c,--config", config_path, "Path to config.json file")->required();
    app.add_option("-s,--safetensors", safetensors_path, "Path to safetensors file")->required();

    CLI11_PARSE(app, argc, argv);

    // Check if files exist
    if (!std::filesystem::exists(config_path)) {
        fmt::println(stderr, "Config file not found: {}", config_path);
        return 1;
    }

    if (!std::filesystem::exists(safetensors_path)) {
        fmt::println(stderr, "SafeTensors file not found: {}", safetensors_path);
        return 1;
    }

    print_config_info(config_path);
    print_safetensors_info(safetensors_path);

    return 0;
}
