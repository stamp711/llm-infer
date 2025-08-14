#include <CLI/CLI.hpp>
#include <iostream>
#include <gguf.hpp>

void print_metadata_value(const MetadataValue& value) {
    std::visit([](const auto& v) {
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
    }, value.inner);
}

int main(int argc, char** argv) {
    CLI::App app{"GGUF file inspector"};
    
    std::string filename;
    app.add_option("file", filename, "GGUF file to inspect")->required();
    
    CLI11_PARSE(app, argc, argv);
    
    try {
        GGUF gguf(filename);
        
        std::cout << "GGUF File Information\n";
        std::cout << "=====================\n";
        std::cout << "File: " << filename << "\n";
        std::cout << "Tensor count: " << gguf.tensor_count() << "\n\n";
        
        std::cout << "Metadata:\n";
        std::cout << "---------\n";
        for (const auto& kv : gguf.metadata_kv()) {
            std::cout << kv.key << ": ";
            print_metadata_value(kv.value);
            std::cout << "\n";
        }
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
