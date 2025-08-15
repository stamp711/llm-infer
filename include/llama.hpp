#pragma once

#include <cstdint>
#include <optional>

struct LlamaConfig {
    std::uint64_t context_length{};
    std::uint64_t embedding_length{};
    std::uint64_t block_count{};
    std::uint64_t feed_forward_length{};
    
    std::uint64_t rope_dimension_count{};
    float rope_freq_base = 10000.0F;
    std::optional<float> rope_scale;
    
    std::uint64_t attention_head_count{};
    std::optional<std::uint64_t> attention_head_count_kv;
    float layer_norm_rms_epsilon{};
    
    std::optional<std::uint32_t> expert_count;
    std::optional<std::uint32_t> expert_used_count;
    
    std::uint64_t vocab_size{};
    
    [[nodiscard]] bool uses_gqa() const noexcept {
        return attention_head_count_kv.has_value() && 
               *attention_head_count_kv != attention_head_count;
    }
    
    [[nodiscard]] bool is_moe() const noexcept {
        return expert_count.has_value() && *expert_count > 0;
    }
    
    [[nodiscard]] std::uint64_t head_dim() const noexcept {
        return embedding_length / attention_head_count;
    }
    
    [[nodiscard]] std::uint64_t kv_head_count() const noexcept {
        return attention_head_count_kv.value_or(attention_head_count);
    }
};