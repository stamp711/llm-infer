#pragma once

#include <cstdint>
#include <filesystem>
#include <mio/mmap.hpp>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "llama.hpp"

// https://github.com/ggml-org/ggml/blob/master/docs/gguf.md

// NOLINTNEXTLINE
enum class GGMLType : std::uint32_t {
    GGML_TYPE_F32 = 0,
    GGML_TYPE_F16 = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    // GGML_TYPE_Q4_2 = 4, support has been removed
    // GGML_TYPE_Q4_3 = 5, support has been removed
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q2_K = 10,
    GGML_TYPE_Q3_K = 11,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_Q8_K = 15,
    GGML_TYPE_IQ2_XXS = 16,
    GGML_TYPE_IQ2_XS = 17,
    GGML_TYPE_IQ3_XXS = 18,
    GGML_TYPE_IQ1_S = 19,
    GGML_TYPE_IQ4_NL = 20,
    GGML_TYPE_IQ3_S = 21,
    GGML_TYPE_IQ2_S = 22,
    GGML_TYPE_IQ4_XS = 23,
    GGML_TYPE_I8 = 24,
    GGML_TYPE_I16 = 25,
    GGML_TYPE_I32 = 26,
    GGML_TYPE_I64 = 27,
    GGML_TYPE_F64 = 28,
    GGML_TYPE_IQ1_M = 29,
    GGML_TYPE_COUNT,
};

// NOLINTNEXTLINE
enum GGUFMetadataValueType : std::uint32_t {
    // The value is a 8-bit unsigned integer.
    GGUF_METADATA_VALUE_TYPE_UINT8 = 0,
    // The value is a 8-bit signed integer.
    GGUF_METADATA_VALUE_TYPE_INT8 = 1,
    // The value is a 16-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT16 = 2,
    // The value is a 16-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT16 = 3,
    // The value is a 32-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT32 = 4,
    // The value is a 32-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT32 = 5,
    // The value is a 32-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT32 = 6,
    // The value is a boolean.
    // 1-byte value where 0 is false and 1 is true.
    // Anything else is invalid, and should be treated as either the model being invalid or the reader being buggy.
    GGUF_METADATA_VALUE_TYPE_BOOL = 7,
    // The value is a UTF-8 non-null-terminated string, with length prepended.
    GGUF_METADATA_VALUE_TYPE_STRING = 8,
    // The value is an array of other values, with the length and type prepended.
    ///
    // Arrays can be nested, and the length of the array is the number of elements in the array, not the number of
    // bytes.
    GGUF_METADATA_VALUE_TYPE_ARRAY = 9,
    // The value is a 64-bit unsigned little-endian integer.
    GGUF_METADATA_VALUE_TYPE_UINT64 = 10,
    // The value is a 64-bit signed little-endian integer.
    GGUF_METADATA_VALUE_TYPE_INT64 = 11,
    // The value is a 64-bit IEEE754 floating point number.
    GGUF_METADATA_VALUE_TYPE_FLOAT64 = 12,
};

struct MetadataValue;
using MetadataArray = std::vector<std::unique_ptr<MetadataValue>>;
struct MetadataValue {
    using VariantType = std::variant<uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float,
                                     double, bool, std::string, MetadataArray>;
    VariantType inner;
};

struct MetadataKeyValue {
    std::string key;
    MetadataValue value;
};

struct TensorInfo {
   public:
    TensorInfo(std::span<const char> &span);

    [[nodiscard]] std::size_t elements() const {
        return std::accumulate(dimensions.begin(), dimensions.end(), 1, std::multiplies<>());
    }

    // The name of the tensor. It is a standard GGUF string, with the caveat that
    // it must be at most 64 bytes long.
    std::string name;
    // The number of dimensions in the tensor.
    // Currently at most 4, but this may change in the future.
    uint32_t n_dimensions;
    // The dimensions of the tensor.
    std::vector<uint64_t> dimensions;
    // The type of the tensor.
    GGMLType type;
    // The offset of the tensor's data in this file in bytes.
    //
    // This offset is relative to `tensor_data`, not to the start
    // of the file, to make it easier for writers to write the file.
    // Readers should consider exposing this offset relative to the
    // file to make it easier to read the data.
    //
    // Must be a multiple of `ALIGNMENT`. That is, `align_offset(offset) == offset`.
    uint64_t offset;

    const std::byte *data = nullptr;
};

class GGUF {
   public:
    GGUF(const std::filesystem::path &path);

    GGUF(const GGUF &) = delete;
    GGUF &operator=(const GGUF &) = delete;
    GGUF(GGUF &&) noexcept = default;
    GGUF &operator=(GGUF &&) noexcept = default;
    ~GGUF() = default;

    [[nodiscard]] std::uint64_t tensor_count() const noexcept { return tensor_count_; }
    [[nodiscard]] const std::vector<MetadataKeyValue> &metadata_kv() const noexcept { return metadata_kv_; }
    [[nodiscard]] const std::unordered_map<std::string, TensorInfo> &tensors() const noexcept { return tensors_; }

    [[nodiscard]] const TensorInfo *get_tensor(const std::string &name) const noexcept {
        auto it = tensors_.find(name);
        return it != tensors_.end() ? &it->second : nullptr;
    }

    [[nodiscard]] const std::string &architecture() const noexcept { return architecture_; }
    [[nodiscard]] std::uint32_t quantization_version() const noexcept { return quantization_version_; }
    [[nodiscard]] std::uint32_t alignment() const noexcept { return alignment_; }

    [[nodiscard]] const MetadataValue *get_metadata(const std::string &key) const noexcept {
        for (const auto &kv : metadata_kv_) {
            if (kv.key == key) {
                return &kv.value;
            }
        }
        return nullptr;
    }

    template <typename T>
    [[nodiscard]] std::optional<T> get_metadata_value(const std::string &key) const noexcept {
        const auto *value = get_metadata(key);
        if (!value) return std::nullopt;

        if (const auto *ptr = std::get_if<T>(&value->inner)) {
            return *ptr;
        }
        return std::nullopt;
    }

    template <typename T>
    [[nodiscard]] T get_metadata_value_or(const std::string &key, T default_value) const noexcept {
        return get_metadata_value<T>(key).value_or(default_value);
    }

    [[nodiscard]] std::optional<LlamaConfig> parse_llama_config() const;

   private:
    mio::mmap_source mmap_;

    // The number of tensors in the file.
    std::uint64_t tensor_count_;
    // The metadata key-value pairs.
    std::vector<MetadataKeyValue> metadata_kv_;  // don't parse value for now, store in bytes

    std::unordered_map<std::string, TensorInfo> tensors_;

    const std::byte *tensor_data_;

    // Required metadata fields
    std::string architecture_;
    std::uint32_t quantization_version_;
    std::uint32_t alignment_;
};
