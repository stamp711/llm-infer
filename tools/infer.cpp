#include <fmt/format.h>

int main() {
    fmt::println("LLM Inference - Use the tools in build/bin/");
    fmt::println("Available tools:");
    fmt::println("  - infer: Main inference tool");
    fmt::println("  - gguf_info: Display GGUF model information and test loading");
    fmt::println("  - hf_info: Display HuggingFace model information");
    return 0;
}