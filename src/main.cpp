#include <CLI/CLI.hpp>
#include <gguf.hpp>
#include <iostream>
#include <string>
#include <tokenizer.hpp>

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
    bool interactive = false;

    app.add_option("file", filename, "GGUF file to load")->required();
    app.add_flag("-m,--metadata", show_metadata, "Show GGUF metadata");
    app.add_flag("-i,--interactive", interactive, "Interactive tokenizer mode");

    CLI11_PARSE(app, argc, argv);

    try {
        std::cout << "Loading GGUF file: " << filename << "\n";
        GGUF gguf(filename);

        std::cout << "File loaded successfully\n";
        std::cout << "Tensor count: " << gguf.tensor_count() << "\n";

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
