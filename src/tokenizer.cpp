#include "tokenizer.hpp"

#include <cstring>
#include <stdexcept>

Tokenizer::Tokenizer(const GGUF& gguf) {
    extract_special_tokens(gguf);

    const auto& metadata = gguf.metadata_kv();

    const MetadataValue* tokens_value = nullptr;
    for (const auto& kv : metadata) {
        if (kv.key == "tokenizer.ggml.tokens") {
            tokens_value = &kv.value;
            break;
        }
    }

    if (tokens_value == nullptr) {
        throw std::runtime_error("No tokenizer.ggml.tokens found in GGUF metadata");
    }

    const auto* tokens_array = std::get_if<MetadataArray>(&tokens_value->inner);
    if (tokens_array == nullptr) {
        throw std::runtime_error("tokenizer.ggml.tokens is not an array");
    }

    vocab_.reserve(tokens_array->size());
    for (const auto& token_ptr : *tokens_array) {
        const auto* token_str = std::get_if<std::string>(&token_ptr->inner);
        if (token_str == nullptr) {
            throw std::runtime_error("Token is not a string");
        }
        vocab_.emplace_back(*token_str);
    }

    for (size_t i = 0; i < vocab_.size(); ++i) {
        const auto& token = vocab_[i];

        if (token == "<0x00>") {
            byte_fallback_start_ = static_cast<std::int32_t>(i);
        }

        if (token == "<|eot_id|>" || token == "<|end|>" || token == "<|im_end|>" || token == "<|endoftext|>" ||
            token == "<|end_of_text|>") {
            eot_id_ = static_cast<std::uint32_t>(i);
        }
    }

    build_trie();
}

void Tokenizer::extract_special_tokens(const GGUF& gguf) {
    const auto& metadata = gguf.metadata_kv();

    for (const auto& kv : metadata) {
        if (kv.key == "tokenizer.ggml.bos_token_id") {
            if (const auto* val = std::get_if<std::uint32_t>(&kv.value.inner)) {
                bos_id_ = *val;
            } else if (const auto* val = std::get_if<std::uint64_t>(&kv.value.inner)) {
                bos_id_ = static_cast<std::uint32_t>(*val);
            }
        } else if (kv.key == "tokenizer.ggml.eos_token_id") {
            if (const auto* val = std::get_if<std::uint32_t>(&kv.value.inner)) {
                eos_id_ = *val;
            } else if (const auto* val = std::get_if<std::uint64_t>(&kv.value.inner)) {
                eos_id_ = static_cast<std::uint32_t>(*val);
            }
        }
    }
}

void Tokenizer::build_trie() {
    for (size_t i = 0; i < vocab_.size(); ++i) {
        const auto& word = vocab_[i];
        TokenTrie* node = &vocab_trie_;

        for (char c : word) {
            if (!node->children.contains(c)) {
                node->children[c] = std::make_unique<TokenTrie>();
            }
            node = node->children[c].get();
        }
        node->token_id = static_cast<std::int32_t>(i);
    }
}

std::vector<std::uint32_t> Tokenizer::encode(std::string_view text, bool add_bos) const {
    std::vector<std::uint32_t> out_tokens;

    // Add BOS token if requested
    if (add_bos) {
        out_tokens.push_back(bos_id_);
    }

    // Process text character by character
    for (size_t i = 0; i < text.size();) {
        size_t longest_match = 0;
        const TokenTrie* best_match = nullptr;

        // Find longest matching token using trie traversal
        const TokenTrie* node = &vocab_trie_;
        for (size_t j = 0; j < text.size() - i; ++j) {
            char c = text[i + j];

            auto it = node->children.find(c);
            if (it == node->children.end()) {
                break;  // No more matches possible
            }

            node = it->second.get();
            if (node->token_id >= 0) {
                best_match = node;
                longest_match = j + 1;
            }
        }

        if (best_match != nullptr) {
            // Use the longest matching token
            out_tokens.push_back(static_cast<std::uint32_t>(best_match->token_id));
            i += longest_match;
        } else {
            // Fallback to byte encoding for unknown characters
            if (byte_fallback_start_.has_value()) {
                auto byte_val = static_cast<unsigned char>(text[i]);
                out_tokens.push_back(static_cast<std::uint32_t>(byte_val + byte_fallback_start_.value()));
            } else {
                throw std::runtime_error("Cannot encode character '" + std::string(1, text[i]) + "' (0x" +
                                         std::to_string(static_cast<unsigned char>(text[i])) +
                                         ") and no byte fallback available");
            }
            i += 1;
        }
    }

    return out_tokens;
}

std::string Tokenizer::decode(const std::vector<std::uint32_t>& tokens) const {
    std::string result;

    for (size_t i = 0; i < tokens.size(); ++i) {
        std::uint32_t prev_token = (i > 0) ? tokens[i - 1] : static_cast<std::uint32_t>(-1);
        result += decode_token_internal_(prev_token, tokens[i]);
    }

    return result;
}

std::string Tokenizer::decode_token(std::uint32_t token) const {
    return decode_token_internal_(static_cast<std::uint32_t>(-1), token);
}

std::string Tokenizer::decode_token_internal_(std::uint32_t prev_token, std::uint32_t token) const {
    // Check token bounds
    if (token >= vocab_.size()) {
        throw std::runtime_error("Token ID " + std::to_string(token) + " out of vocab bounds");
    }

    const std::string& piece = vocab_[token];

    // Strip leading whitespace after BOS token (SentencePiece behavior)
    if (prev_token == bos_id_ && !piece.empty() && piece[0] == ' ') {
        return piece.substr(1);
    }

    // Handle byte fallback tokens
    if (byte_fallback_start_.has_value()) {
        auto start = byte_fallback_start_.value();
        auto end = start + 256;

        if (token >= start && token < end) {
            auto byte = static_cast<char>(token - start);
            return {byte};
        }
    }

    return piece;
}

std::string Tokenizer::tokens_to_debug_string(const std::vector<std::uint32_t>& tokens) const {
    std::string result;

    for (std::uint32_t token : tokens) {
        if (token == bos_id_) {
            result += "[<s>:" + std::to_string(token) + "]";
        } else if (token == eos_id_) {
            result += "[</s>:" + std::to_string(token) + "]";
        } else if (eot_id_.has_value() && token == eot_id_.value()) {
            result += "[<eot>:" + std::to_string(token) + "]";
        } else if (token < vocab_.size()) {
            result += "[" + vocab_[token] + ":" + std::to_string(token) + "]";
        } else {
            result += "[INVALID:" + std::to_string(token) + "]";
        }
    }

    return result;
}
