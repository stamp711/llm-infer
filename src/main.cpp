#include <CLI/CLI.hpp>
#include <gguf.hpp>
#include <iostream>
#include <magic_enum/magic_enum.hpp>
#include <string>
#include <tokenizer.hpp>

void print_tensor_info(const std::vector<TensorInfo>& tensor_infos) {
    std::cout << "\nTensor Information:\n";
    std::cout << "==================\n\n";

    for (size_t i = 0; i < tensor_infos.size(); ++i) {
        const auto& tensor = tensor_infos[i];
        std::cout << "Tensor #" << i << ":\n";
        std::cout << "  Name: " << tensor.name << "\n";
        std::cout << "  Type: " << magic_enum::enum_name(tensor.type) << "\n";
        std::cout << "  Shape: [";
        for (size_t j = 0; j < tensor.dimensions.size(); ++j) {
            std::cout << tensor.dimensions[j];
            if (j < tensor.dimensions.size() - 1) std::cout << ", ";
        }
        std::cout << "]\n";
        std::cout << "  Offset: " << tensor.offset << " bytes\n";
        std::cout << "  Data address: " << static_cast<const void*>(tensor.data) << "\n";

        uint64_t total_elements = 1;
        for (auto dim : tensor.dimensions) {
            total_elements *= dim;
        }
        std::cout << "  Elements: " << total_elements << "\n\n";
    }

    std::cout << "Total tensors: " << tensor_infos.size() << "\n";
}

void print_metadata_value(const MetadataValue& value) {
    std::visit(
        [](const auto& v) {
            using T = std::decay_t<decltype(v)>;
            if constexpr (std::is_same_v<T, MetadataArray>) {
                std::cout << "[array of " << v.size() << " elements]";
            } else if constexpr (std::is_same_v<T, std::string>) {
                std::cout << "\"" << v << "\"";
            } else if constexpr (std::is_same_v<T, bool>) {
                std::cout << (v ? "true" : "false");
            } else {
                std::cout << v;
            }
        },
        value.inner);
}

void test_tokenizer(const Tokenizer& tokenizer) {
    std::cout << "\nTokenizer Test\n";
    std::cout << "==============\n";
    std::cout << "Vocab size: " << tokenizer.vocab_size() << "\n";
    std::cout << "BOS token: " << tokenizer.bos_token() << "\n";
    std::cout << "EOS token: " << tokenizer.eos_token() << "\n";

    if (auto eot = tokenizer.eot_token()) {
        std::cout << "EOT token: " << eot.value() << "\n";
    }

    std::cout << "> ";

    std::string input;
    while (std::getline(std::cin, input)) {
        if (input.empty()) {
            continue;
        }

        try {
            // Encode the input
            auto tokens = tokenizer.encode(input, true);

            std::cout << "Tokens: " << tokenizer.tokens_to_debug_string(tokens) << "\n";
            std::cout << "Token count: " << tokens.size() << "\n";

            // Decode back
            std::string decoded = tokenizer.decode(tokens);
            std::cout << "Decoded: \"" << decoded << "\"\n";

            // Check decoded string
            auto expected = tokenizer.decode({tokenizer.bos_token()}) + input;
            if (decoded != expected) {
                std::cout << "Difference detected\n";
                std::cout << "Original: \"" << input << "\"\n";
                std::cout << "Expected: \"" << expected << "\"\n";
                std::cout << "Decoded:  \"" << decoded << "\"\n";
            }

        } catch (const std::exception& e) {
            std::cout << "Error: " << e.what() << "\n";
        }

        std::cout << "> ";
    }
}

int main(int argc, char** argv) {
    CLI::App app{"LLM Inference with Tokenizer"};

    std::string filename;
    bool show_metadata = false;
    bool show_tensors = false;
    bool interactive = false;

    app.add_option("file", filename, "GGUF file to load")->required();
    app.add_flag("-m,--metadata", show_metadata, "Show GGUF metadata");
    app.add_flag("-t,--tensors", show_tensors, "Show tensor information");
    app.add_flag("-i,--interactive", interactive, "Interactive tokenizer mode");

    CLI11_PARSE(app, argc, argv);

    try {
        std::cout << "Loading GGUF file: " << filename << "\n";
        GGUF gguf(filename);

        std::cout << "File loaded successfully\n";
        std::cout << "Architecture: " << gguf.architecture() << "\n";
        std::cout << "Quantization version: " << gguf.quantization_version() << "\n";
        std::cout << "Alignment: " << gguf.alignment() << " bytes\n";
        std::cout << "Tensor count: " << gguf.tensor_count() << "\n";
        
        if (gguf.architecture() == "llama") {
            if (auto llama_config = gguf.parse_llama_config()) {
                std::cout << "\nLlama Model Configuration:\n";
                std::cout << "  Context length: " << llama_config->context_length << "\n";
                std::cout << "  Embedding size: " << llama_config->embedding_length << "\n";
                std::cout << "  Block count: " << llama_config->block_count << "\n";
                std::cout << "  Feed forward size: " << llama_config->feed_forward_length << "\n";
                std::cout << "  Attention heads: " << llama_config->attention_head_count << "\n";
                std::cout << "  KV heads: " << llama_config->kv_head_count() << "\n";
                std::cout << "  Head dimension: " << llama_config->head_dim() << "\n";
                std::cout << "  RoPE dimension: " << llama_config->rope_dimension_count << "\n";
                std::cout << "  RoPE freq base: " << llama_config->rope_freq_base << "\n";
                std::cout << "  RMS norm epsilon: " << llama_config->layer_norm_rms_epsilon << "\n";
                std::cout << "  Vocab size: " << llama_config->vocab_size << "\n";
                std::cout << "  Uses GQA: " << (llama_config->uses_gqa() ? "yes" : "no") << "\n";
                std::cout << "  Is MoE: " << (llama_config->is_moe() ? "yes" : "no") << "\n";
            } else {
                std::cout << "Failed to parse Llama configuration\n";
            }
        }

        // Initialize tokenizer
        std::cout << "Initializing tokenizer...\n";
        Tokenizer tokenizer(gguf);
        std::cout << "Tokenizer initialized\n";

        if (show_metadata) {
            std::cout << "\nGGUF Metadata:\n";
            std::cout << "==============\n";
            for (const auto& kv : gguf.metadata_kv()) {
                std::cout << kv.key << ": ";
                print_metadata_value(kv.value);
                std::cout << "\n";
            }
        }

        if (show_tensors) {
            print_tensor_info(gguf.tensor_infos());
        }

        if (interactive) {
            test_tokenizer(tokenizer);
        } else {
            std::cout << "\nTokenizer ready. Use -i flag for interactive mode.\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }

    return 0;
}
