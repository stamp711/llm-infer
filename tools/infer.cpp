#include <fmt/format.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gguf.hpp"
#include "model/device.hpp"
#include "model/inference_state.hpp"
#include "model/model.hpp"
#include "tokenizer.hpp"

// ============================================================================
// Timing Utilities
// ============================================================================

std::uint64_t get_timestamp_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::high_resolution_clock::now().time_since_epoch())
        .count();
}

// ============================================================================
// Sampler Implementation
// ============================================================================

class Sampler {
   public:
    explicit Sampler(std::uint32_t vocab_size, std::uint64_t seed = 0)
        : vocab_size_(vocab_size), rng_(seed == 0 ? std::random_device{}() : seed) {}

    // Sample token using temperature
    std::uint32_t sample(const float* logits, float temperature = 1.0F) {
        if (temperature <= 0.0F) {
            return sample_argmax(logits);
        }

        // Apply temperature and compute softmax
        std::vector<float> probs(vocab_size_);
        softmax(probs.data(), logits, temperature);

        // Sample from distribution
        std::uniform_real_distribution<float> dist(0.0F, 1.0F);
        float r = dist(rng_);

        float cumsum = 0.0F;
        for (std::uint32_t i = 0; i < vocab_size_; ++i) {
            cumsum += probs[i];
            if (r < cumsum) {
                return i;
            }
        }
        return vocab_size_ - 1;  // fallback
    }

    // Deterministic sampling (argmax)
    std::uint32_t sample_argmax(const float* logits) const {
        std::uint32_t max_i = 0;
        float max_val = logits[0];
        for (std::uint32_t i = 1; i < vocab_size_; ++i) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_i = i;
            }
        }
        return max_i;
    }

    // Get probability of specific token
    float sample_prob(std::uint32_t token_id, const float* logits) {
        std::vector<float> probs(vocab_size_);
        softmax(probs.data(), logits, 1.0F);
        return (token_id < vocab_size_) ? probs[token_id] : 0.0F;
    }

   private:
    void softmax(float* output, const float* input, float temperature) const {
        // Find max for numerical stability
        float max_val = input[0];
        for (std::uint32_t i = 1; i < vocab_size_; ++i) {
            max_val = std::max(max_val, input[i]);
        }

        // Compute exp and sum
        float sum = 0.0F;
        for (std::uint32_t i = 0; i < vocab_size_; ++i) {
            output[i] = std::exp((input[i] - max_val) / temperature);
            sum += output[i];
        }

        // Normalize
        for (std::uint32_t i = 0; i < vocab_size_; ++i) {
            output[i] /= sum;
        }
    }

    std::uint32_t vocab_size_;
    std::mt19937 rng_;
};

// ============================================================================
// CLI Utilities
// ============================================================================

void print_usage() {
    fmt::println("Usage: infer <model.gguf> [options]");
    fmt::println("Example: infer model.gguf -i \"Hello, world!\"");
    fmt::println("");
    fmt::println("Options:");
    fmt::println("  -h                    Display this help message");
    fmt::println("  -d <device>           Device to use: cpu, cuda (default: cpu)");
    fmt::println("  -i <string>           Input prompt");
    fmt::println("  -f <file>             Input prompt from file");
    fmt::println("  -n <int>              Number of tokens to generate (default: 256)");
    fmt::println("  -t <float>            Temperature for sampling (default: 1.0)");
    fmt::println("  -T <int>              Context length override (default: model max)");
    fmt::println("  --seed <int>          Random seed (default: random)");
}

struct Config {
    std::string model_path;
    std::string device = "cpu";
    std::string prompt;
    std::string prompt_file;
    int max_tokens = 256;
    float temperature = 1.0F;
    std::uint64_t seed = 0;
    bool show_help = false;
};

bool parse_args(int argc, char** argv, Config& config) {
    // Check for help first, before requiring model path
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "-h" || std::string(argv[i]) == "--help") {
            config.show_help = true;
            return true;
        }
    }

    if (argc < 2) {
        return false;
    }

    config.model_path = argv[1];

    for (int i = 2; i < argc;) {
        std::string arg = argv[i];

        if (arg == "-h" || arg == "--help") {
            config.show_help = true;
            return true;
        }
        if (arg == "-d") {
            if (i + 1 >= argc) return false;
            config.device = argv[i + 1];
            i += 2;
        } else if (arg == "-i") {
            if (i + 1 >= argc) return false;
            config.prompt = argv[i + 1];
            i += 2;
        } else if (arg == "-f") {
            if (i + 1 >= argc) return false;
            config.prompt_file = argv[i + 1];
            i += 2;
        } else if (arg == "-n") {
            if (i + 1 >= argc) return false;
            config.max_tokens = std::stoi(argv[i + 1]);
            i += 2;
        } else if (arg == "-t") {
            if (i + 1 >= argc) return false;
            config.temperature = std::stof(argv[i + 1]);
            i += 2;
        } else if (arg == "--seed") {
            if (i + 1 >= argc) return false;
            config.seed = std::stoull(argv[i + 1]);
            i += 2;
        } else {
            fmt::println(stderr, "Unknown argument: {}", arg);
            return false;
        }
    }

    // Validate device
    if (config.device != "cpu" && config.device != "cuda") {
        fmt::println(stderr, "Invalid device: {}. Must be 'cpu' or 'cuda'", config.device);
        return false;
    }

    // Must have either prompt or prompt file (but not for help)
    if (!config.show_help && config.prompt.empty() && config.prompt_file.empty()) {
        fmt::println(stderr, "Must specify either -i <prompt> or -f <file>");
        return false;
    }

    // Can't have both
    if (!config.prompt.empty() && !config.prompt_file.empty()) {
        fmt::println(stderr, "Cannot specify both -i and -f");
        return false;
    }

    return true;
}

// ========== Main Inference Function ==========

void run_inference(const Config& config) {
    try {
        // Load model
        fmt::println("Loading model: {}", config.model_path);
        GGUF gguf(config.model_path);

        DeviceType device_type = (config.device == "cuda") ? DeviceType::CUDA : DeviceType::CPU;
        Model model(gguf, device_type);
        const auto& model_config = model.config();

        // Create inference state and tokenizer
        InferenceState state(model_config, device_type);
        Tokenizer tokenizer(gguf);
        Sampler sampler(model_config.vocab_size, config.seed);

        fmt::println("Model load successful");
        fmt::println("  Layers: {}", model_config.n_layers);
        fmt::println("  Dimensions: {}", model_config.dim);
        fmt::println("  Heads: {}", model_config.n_heads);
        fmt::println("  KV Heads: {}", model_config.n_kv_heads);
        fmt::println("  Vocab size: {}", model_config.vocab_size);
        fmt::println("  Max seq len: {}", model_config.max_seq_len);
        fmt::println("  Device: {}", config.device);
        fmt::println("");

        // Load prompt
        std::string prompt_text = config.prompt;
        if (!config.prompt_file.empty()) {
            std::ifstream file(config.prompt_file);
            if (!file.is_open()) {
                throw std::runtime_error("Failed to open prompt file: " + config.prompt_file);
            }
            std::stringstream buffer;
            buffer << file.rdbuf();
            prompt_text = buffer.str();
        }

        // Warmup
        fmt::println("Warming up...");
        model.forward(state, 0, 0, InferenceMode::OutputLogits);

        // Encode prompt
        fmt::println("Encoding prompt...");
        auto tokens = tokenizer.encode(prompt_text, true);
        fmt::println("Encoded {} tokens", tokens.size());
        fmt::println("Token debug: {}", tokenizer.tokens_to_debug_string(tokens));
        fmt::println("");

        if (tokens.size() >= model_config.max_seq_len) {
            throw std::runtime_error(
                fmt::format("Prompt too long: {} tokens (max: {})", tokens.size(), model_config.max_seq_len));
        }

        // Prefill phase - process all prompt tokens
        fmt::println("Prefilling...");
        std::uint64_t prefill_start = get_timestamp_ms();

        for (std::size_t pos = 0; pos < tokens.size(); ++pos) {
            InferenceMode mode =
                (pos + 1 == tokens.size()) ? InferenceMode::OutputLogits : InferenceMode::HydrateKVCache;
            model.forward(state, tokens[pos], static_cast<std::uint32_t>(pos), mode);
        }

        std::uint64_t prefill_end = get_timestamp_ms();
        double prefill_time_s = static_cast<double>(prefill_end - prefill_start) / 1000.0;
        fmt::println("Prefill completed in {:.3f}s ({:.1f} tok/s)", prefill_time_s,
                     static_cast<double>(tokens.size()) / prefill_time_s);
        fmt::println("");

        // Generation phase
        fmt::println("Generating...");
        std::uint64_t generation_start = get_timestamp_ms();

        std::cout << "Output: " << std::flush;

        std::uint32_t generated_tokens = 0;
        for (int i = 0; i < config.max_tokens; ++i) {
            // Sample next token
            std::uint32_t next_token = sampler.sample(state.logits(), config.temperature);

            // Check for EOS
            auto eot_token = tokenizer.eot_token();
            if (next_token == tokenizer.eos_token() || (eot_token.has_value() && next_token == *eot_token)) {
                break;
            }

            // Decode and print
            std::uint32_t prev_token = tokens.back();
            std::string token_str = tokenizer.decode_token(next_token);

            // Handle special decoding (like removing leading space after BOS)
            if (prev_token == tokenizer.bos_token() && !token_str.empty() && token_str[0] == ' ') {
                token_str = token_str.substr(1);
            }

            // Convert SentencePiece ▁ to spaces for display only
            size_t pos = 0;
            while ((pos = token_str.find("▁", pos)) != std::string::npos) {
                token_str.replace(pos, 3, " ");  // ▁ is 3 bytes in UTF-8
                pos += 1;
            }

            std::cout << token_str << std::flush;
            tokens.push_back(next_token);
            generated_tokens++;

            // Forward for next iteration
            if (i < config.max_tokens - 1) {  // Don't forward on last iteration
                auto pos = static_cast<std::uint32_t>(tokens.size() - 1);
                model.forward(state, next_token, pos, InferenceMode::OutputLogits);
            }
        }

        std::cout << '\n' << '\n';

        std::uint64_t generation_end = get_timestamp_ms();
        double generation_time_s = static_cast<double>(generation_end - generation_start) / 1000.0;
        double total_time_s = static_cast<double>(generation_end - prefill_start) / 1000.0;

        // Print statistics
        fmt::println("Generation Statistics:");
        fmt::println("  Prompt tokens: {}", tokens.size() - generated_tokens);
        fmt::println("  Generated tokens: {}", generated_tokens);
        fmt::println("  Total tokens: {}", tokens.size());
        fmt::println("  Prefill time: {:.3f}s ({:.1f} tok/s)", prefill_time_s,
                     static_cast<double>(tokens.size() - generated_tokens) / prefill_time_s);
        fmt::println("  Generation time: {:.3f}s ({:.1f} tok/s)", generation_time_s,
                     static_cast<double>(generated_tokens) / generation_time_s);
        fmt::println("  Total time: {:.3f}s ({:.1f} tok/s)", total_time_s,
                     static_cast<double>(tokens.size()) / total_time_s);
        fmt::println("  Temperature: {:.2f}", config.temperature);

    } catch (const std::exception& e) {
        fmt::println(stderr, "Error: {}", e.what());
        std::exit(1);
    }
}

// ============================================================================
// Main Entry Point
// ============================================================================

int main(int argc, char** argv) {
    Config config;

    if (!parse_args(argc, argv, config)) {
        print_usage();
        return 1;
    }

    if (config.show_help) {
        print_usage();
        return 0;
    }

    run_inference(config);
    return 0;
}
