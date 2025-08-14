#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "gguf.hpp"

struct TokenTrie {
    std::unordered_map<char, std::unique_ptr<TokenTrie>> children;
    std::int32_t token_id = -1;
};

class Tokenizer {
   public:
    explicit Tokenizer(const GGUF& gguf);

    [[nodiscard]] std::vector<std::uint32_t> encode(std::string_view text, bool add_bos = true) const;

    [[nodiscard]] std::string decode(const std::vector<std::uint32_t>& tokens) const;
    [[nodiscard]] std::string decode_token(std::uint32_t token) const;

    [[nodiscard]] std::size_t vocab_size() const noexcept { return vocab_.size(); }

    [[nodiscard]] std::uint32_t bos_token() const noexcept { return bos_id_; }
    [[nodiscard]] std::uint32_t eos_token() const noexcept { return eos_id_; }
    [[nodiscard]] std::optional<std::uint32_t> eot_token() const noexcept { return eot_id_; }

    [[nodiscard]] std::string tokens_to_debug_string(const std::vector<std::uint32_t>& tokens) const;

    [[nodiscard]] std::optional<std::uint32_t> byte_fallback_start() const noexcept { return byte_fallback_start_; }

   private:
    void build_trie();
    void extract_special_tokens(const GGUF& gguf);
    [[nodiscard]] std::string decode_token_internal_(std::uint32_t prev_token, std::uint32_t token) const;

    std::vector<std::string> vocab_;
    TokenTrie vocab_trie_;

    std::uint32_t bos_id_ = 1;
    std::uint32_t eos_id_ = 2;
    std::optional<std::uint32_t> eot_id_;

    std::optional<std::uint32_t> byte_fallback_start_;
};
