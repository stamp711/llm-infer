#include "gguf.hpp"

#include <stdexcept>
#include <utility>

constexpr uint32_t GGUF_MAGIC = 0x46554747;

namespace {

struct __attribute__((packed)) GGUFHeaderWithoutMetadata {
    // Magic number to announce that this is a GGUF file.
    // Must be `GGUF` at the byte level: `0x47` `0x47` `0x55` `0x46`.
    // Your executor might do little-endian byte order, so it might be
    // check for 0x46554747 and letting the endianness cancel out.
    // Consider being *very* explicit about the byte order here.
    uint32_t magic;
    // The version of the format implemented.
    // Must be `3` for version described in this spec, which introduces big-endian support.
    //
    // This version should only be increased for structural changes to the format.
    // Changes that do not affect the structure of the file should instead update the metadata
    // to signify the change.
    uint32_t version;
    // The number of tensors in the file.
    // This is explicit, instead of being included in the metadata, to ensure it is always present
    // for loading the tensors.
    uint64_t tensor_count;
    // The number of metadata key-value pairs.
    uint64_t metadata_kv_count;
    // The metadata key-value pairs.
    // gguf_metadata_kv_t metadata_kv[metadata_kv_count];
};

std::string parse_gguf_string(std::span<const char>& span) {
    std::uint64_t length = *reinterpret_cast<const std::uint64_t*>(span.data());
    span = span.subspan(sizeof(length));
    const char* str = span.data();
    span = span.subspan(length);
    return {str, length};
}

MetadataValue parse_metadata_value(std::span<const char>& span,
                                   std::optional<GGUFMetadataValueType> type = std::nullopt) {
    if (!type) {
        type = GGUFMetadataValueType{*reinterpret_cast<const std::uint32_t*>(span.data())};
        span = span.subspan(sizeof(std::uint32_t));
    }
    switch (*type) {
        case GGUF_METADATA_VALUE_TYPE_UINT8: {
            auto value = *reinterpret_cast<const std::uint8_t*>(span.data());
            span = span.subspan(1);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_INT8: {
            auto value = *reinterpret_cast<const std::int8_t*>(span.data());
            span = span.subspan(1);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_UINT16: {
            auto value = *reinterpret_cast<const std::uint16_t*>(span.data());
            span = span.subspan(2);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_INT16: {
            auto value = *reinterpret_cast<const std::int16_t*>(span.data());
            span = span.subspan(2);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_UINT32: {
            auto value = *reinterpret_cast<const std::uint32_t*>(span.data());
            span = span.subspan(4);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_INT32: {
            auto value = *reinterpret_cast<const std::int32_t*>(span.data());
            span = span.subspan(4);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_UINT64: {
            auto value = *reinterpret_cast<const std::uint64_t*>(span.data());
            span = span.subspan(8);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_INT64: {
            auto value = *reinterpret_cast<const std::int64_t*>(span.data());
            span = span.subspan(8);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_FLOAT64: {
            auto value = *reinterpret_cast<const double*>(span.data());
            span = span.subspan(8);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_FLOAT32: {
            auto value = *reinterpret_cast<const float*>(span.data());
            span = span.subspan(4);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_BOOL: {  // 1-byte value
            bool value = static_cast<bool>(*span.data());
            span = span.subspan(1);
            return {value};
        }
        case GGUF_METADATA_VALUE_TYPE_STRING: {
            return {parse_gguf_string(span)};
        }
        case GGUF_METADATA_VALUE_TYPE_ARRAY: {
            auto type = GGUFMetadataValueType{*reinterpret_cast<const std::uint32_t*>(span.data())};
            span = span.subspan(4);

            uint64_t length = *reinterpret_cast<const uint64_t*>(span.data());
            span = span.subspan(8);

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

// TensorInfo TensorInfo::parse(const std::byte* data) {}

GGUF::GGUF(const std::filesystem::path& path) : mmap_(path.c_str()) {
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

    tensor_data_ = span.data();
}
