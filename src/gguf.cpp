#include "gguf.hpp"

#include <stdexcept>
#include <utility>

namespace {

std::string parse_gguf_string(std::span<const char>& span) {
    std::uint64_t length = *reinterpret_cast<const std::uint64_t*>(span.data());
    span = span.subspan(sizeof(length));
    const char* str = span.data();
    span = span.subspan(length);
    return {str, length};
}

template <typename T>
T parse(std::span<const char>& span) {
    T value = *reinterpret_cast<const T*>(span.data());
    span = span.subspan(sizeof(T));
    return value;
}

constexpr auto parse_uint8 = parse<std::uint8_t>;
constexpr auto parse_uint16 = parse<std::uint16_t>;
constexpr auto parse_uint32 = parse<std::uint32_t>;
constexpr auto parse_uint64 = parse<std::uint64_t>;

constexpr auto parse_int8 = parse<std::int8_t>;
constexpr auto parse_int16 = parse<std::int16_t>;
constexpr auto parse_int32 = parse<std::int32_t>;
constexpr auto parse_int64 = parse<std::int64_t>;

constexpr auto parse_float32 = parse<float>;
constexpr auto parse_float64 = parse<double>;

GGUFMetadataValueType parse_gguf_metadata_value_type(std::span<const char>& span) {
    return GGUFMetadataValueType{parse_uint32(span)};
}
GGMLType parse_ggml_type(std::span<const char>& span) { return GGMLType{parse_uint32(span)}; }

MetadataValue parse_metadata_value(std::span<const char>& span,
                                   std::optional<GGUFMetadataValueType> type = std::nullopt) {
    if (!type) type = parse_gguf_metadata_value_type(span);
    switch (*type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8: return {parse_uint8(span)};
        case GGUF_METADATA_VALUE_TYPE_INT8: return {parse_int8(span)};
        case GGUF_METADATA_VALUE_TYPE_UINT16: return {parse_uint16(span)};
        case GGUF_METADATA_VALUE_TYPE_INT16: return {parse_int16(span)};
        case GGUF_METADATA_VALUE_TYPE_UINT32: return {parse_uint32(span)};
        case GGUF_METADATA_VALUE_TYPE_INT32: return {parse_int32(span)};
        case GGUF_METADATA_VALUE_TYPE_UINT64: return {parse_uint64(span)};
        case GGUF_METADATA_VALUE_TYPE_INT64: return {parse_int64(span)};
        case GGUF_METADATA_VALUE_TYPE_FLOAT64: return {parse_float64(span)};
        case GGUF_METADATA_VALUE_TYPE_FLOAT32: return {parse_float32(span)};
        case GGUF_METADATA_VALUE_TYPE_BOOL: return {static_cast<bool>(parse_uint8(span))};  // 1-byte value
        case GGUF_METADATA_VALUE_TYPE_STRING: return {parse_gguf_string(span)};
        case GGUF_METADATA_VALUE_TYPE_ARRAY: {
            auto type = parse_gguf_metadata_value_type(span);
            auto length = parse_uint64(span);
            MetadataArray values;
            values.reserve(length);
            for (uint64_t i = 0; i < length; ++i) {
                values.emplace_back(std::make_unique<MetadataValue>(parse_metadata_value(span, type)));
            }
            return {std::move(values)};
        }
    }
    throw std::runtime_error("Unknown metadata value type");
}

MetadataKeyValue parse_metadata_kv(std::span<const char>& span) {
    auto key = parse_gguf_string(span);
    auto value = parse_metadata_value(span);
    return {.key = std::move(key), .value = std::move(value)};
}

}  // namespace

TensorInfo::TensorInfo(std::span<const char>& span) : name(parse_gguf_string(span)), n_dimensions(parse_uint32(span)) {
    for (uint32_t i = 0; i < n_dimensions; ++i) {
        dimensions.push_back(parse_uint64(span));
    }
    type = parse_ggml_type(span);
    offset = parse_uint64(span);
}

std::optional<LlamaConfig> GGUF::parse_llama_config() const {
    if (architecture_ != "llama") {
        return std::nullopt;
    }

    LlamaConfig config;

    auto get_required = [this](const std::string& key) -> std::optional<std::uint64_t> {
        if (auto value = get_metadata_value<std::uint64_t>("llama." + key)) {
            return value;
        }
        if (auto value = get_metadata_value<std::uint32_t>("llama." + key)) {
            return static_cast<std::uint64_t>(*value);
        }
        return std::nullopt;
    };

    auto get_required_float = [this](const std::string& key) -> std::optional<float> {
        if (auto value = get_metadata_value<float>("llama." + key)) {
            return value;
        }
        if (auto value = get_metadata_value<double>("llama." + key)) {
            return static_cast<float>(*value);
        }
        return std::nullopt;
    };

    auto context_length = get_required("context_length");
    if (!context_length) return std::nullopt;
    config.context_length = *context_length;

    auto embedding_length = get_required("embedding_length");
    if (!embedding_length) return std::nullopt;
    config.embedding_length = *embedding_length;

    auto block_count = get_required("block_count");
    if (!block_count) return std::nullopt;
    config.block_count = *block_count;

    auto feed_forward_length = get_required("feed_forward_length");
    if (!feed_forward_length) return std::nullopt;
    config.feed_forward_length = *feed_forward_length;

    auto rope_dimension_count = get_required("rope.dimension_count");
    if (!rope_dimension_count) return std::nullopt;
    config.rope_dimension_count = *rope_dimension_count;

    auto attention_head_count = get_required("attention.head_count");
    if (!attention_head_count) return std::nullopt;
    config.attention_head_count = *attention_head_count;

    auto layer_norm_rms_epsilon = get_required_float("attention.layer_norm_rms_epsilon");
    if (!layer_norm_rms_epsilon) return std::nullopt;
    config.layer_norm_rms_epsilon = *layer_norm_rms_epsilon;

    if (auto rope_freq_base = get_required_float("rope.freq_base")) {
        config.rope_freq_base = *rope_freq_base;
    }

    if (auto rope_scale = get_required_float("rope.scale")) {
        config.rope_scale = rope_scale;
    }

    if (auto head_count_kv = get_required("attention.head_count_kv")) {
        config.attention_head_count_kv = head_count_kv;
    }

    if (auto expert_count = get_metadata_value<std::uint32_t>("llama.expert_count")) {
        config.expert_count = expert_count;
    }

    if (auto expert_used = get_metadata_value<std::uint32_t>("llama.expert_used_count")) {
        config.expert_used_count = expert_used;
    }

    if (auto vocab_size = get_required("vocab_size")) {
        config.vocab_size = *vocab_size;
    } else {
        auto tokenizer_vocab_size = get_metadata_value<std::uint32_t>("tokenizer.ggml.model");
        if (!tokenizer_vocab_size) return std::nullopt;
        config.vocab_size = *tokenizer_vocab_size;
    }

    return config;
}

GGUF::GGUF(const std::filesystem::path& path) : mmap_(path.c_str()) {
    constexpr std::uint32_t GGUF_MAGIC = 0x46554747;
    struct __attribute__((packed)) GGUFHeaderWithoutMetadata {
        // Magic number to announce that this is a GGUF file.
        // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
        // Your executor might do little-endian byte order, so it might be
        // check for 0x46554747 and letting the endianness cancel out.
        // Consider being *very* explicit about the byte order here.
        std::uint32_t magic;
        // The version of the format implemented.
        // Must be `3` for version described in this spec, which introduces big-endian support.
        std::uint32_t version;
        // The number of tensors in the file.
        // This is explicit, instead of being included in the metadata, to ensure it is always present
        // for loading the tensors.
        std::uint64_t tensor_count;
        // The number of metadata key-value pairs.
        std::uint64_t metadata_kv_count;
        // The metadata key-value pairs.
        // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
    };

    if (mmap_.size() < sizeof(GGUFHeaderWithoutMetadata)) {
        throw std::runtime_error("Invalid GGUF file");
    }

    // Check magic number and version
    const auto* header = reinterpret_cast<const GGUFHeaderWithoutMetadata*>(mmap_.data());
    if (header->magic != GGUF_MAGIC) {
        throw std::runtime_error("Invalid GGUF file");
    }
    if (header->version != 3) {
        throw std::runtime_error("Unsupported GGUF version");
    }

    tensor_count_ = header->tensor_count;

    // Parse metadata key-value pairs
    std::span<const char> span(mmap_);
    span = span.subspan(sizeof(GGUFHeaderWithoutMetadata));
    for (size_t i = 0; i < header->metadata_kv_count; ++i) {
        auto kv = parse_metadata_kv(span);
        metadata_kv_.push_back(std::move(kv));
    }

    // Parse required metadata fields
    if (const auto* arch = get_metadata("general.architecture")) {
        architecture_ = std::get<std::string>(arch->inner);
    } else {
        throw std::runtime_error("Missing required metadata: general.architecture");
    }

    if (const auto* quant_ver = get_metadata("general.quantization_version")) {
        quantization_version_ = std::get<uint32_t>(quant_ver->inner);
    } else {
        throw std::runtime_error("Missing required metadata: general.quantization_version");
    }

    if (const auto* align = get_metadata("general.alignment")) {
        alignment_ = std::get<uint32_t>(align->inner);
    } else {
        // If general.alignment is not present, use default value of 32
        alignment_ = 32;
    }

    // Parse tensor info
    for (size_t i = 0; i < header->tensor_count; ++i) {
        TensorInfo tensor_info(span);
        tensors_.emplace(tensor_info.name, std::move(tensor_info));
    }

    // Calculate and skip padding
    if (alignment_ % 8 != 0) {
        throw std::runtime_error("Invalid alignment value " + std::to_string(alignment_));
    }
    auto diff = span.data() - mmap_.data();
    auto padding = (alignment_ - diff % alignment_) % alignment_;
    span = span.subspan(padding);
    tensor_data_ = reinterpret_cast<const std::byte*>(span.data());

    // Update tensor addresses
    for (auto& [name, tensor] : tensors_) {
        tensor.data = tensor_data_ + tensor.offset;
    }
}
