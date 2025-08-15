#pragma once

#include <cstdint>
#include <filesystem>
#include <mio/mmap.hpp>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace safetensors {

struct string_transparent_hash {
    using is_transparent = void;
    std::size_t operator()(std::string_view str) const noexcept { return std::hash<std::string_view>{}(str); }
    std::size_t operator()(const std::string& str) const noexcept { return std::hash<std::string>{}(str); }
    std::size_t operator()(const char* str) const noexcept { return std::hash<std::string_view>{}(str); }
};

class SafeTensorException : public std::runtime_error {
   public:
    explicit SafeTensorException(const std::string& message) : std::runtime_error(message) {}
};

/// The various available dtypes. They MUST be in increasing alignment order
enum class Dtype : std::uint8_t {
    /// Boolean type
    BOOL = 0,
    /// MXF4 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F4,
    /// MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F6_E2M3,
    /// MXF6 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F6_E3M2,
    /// Unsigned byte
    U8,
    /// Signed byte
    I8,
    /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    F8_E5M2,
    /// FP8 <https://arxiv.org/pdf/2209.05433.pdf>_
    F8_E4M3,
    /// F8_E8M0 <https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf>_
    F8_E8M0,
    /// Signed integer (16-bit)
    I16,
    /// Unsigned integer (16-bit)
    U16,
    /// Half-precision floating point
    F16,
    /// Brain floating point
    BF16,
    /// Signed integer (32-bit)
    I32,
    /// Unsigned integer (32-bit)
    U32,
    /// Floating point (32-bit)
    F32,
    /// Floating point (64-bit)
    F64,
    /// Signed integer (64-bit)
    I64,
    /// Unsigned integer (64-bit)
    U64,
};

/// Gives out the size (in bits) of 1 element of this dtype.
static constexpr std::size_t dtype_bitsize(Dtype dtype) {
    switch (dtype) {
        case Dtype::F4: return 4;
        case Dtype::F6_E3M2:
        case Dtype::F6_E2M3: return 6;
        case Dtype::BOOL:
        case Dtype::U8:
        case Dtype::I8:
        case Dtype::F8_E5M2:
        case Dtype::F8_E4M3:
        case Dtype::F8_E8M0: return 8;
        case Dtype::I16:
        case Dtype::U16:
        case Dtype::F16:
        case Dtype::BF16: return 16;
        case Dtype::I32:
        case Dtype::U32:
        case Dtype::F32: return 32;
        case Dtype::I64:
        case Dtype::U64:
        case Dtype::F64: return 64;
    }
    return 8;  // unreachable - all enum cases covered
}

/// A single tensor information.
/// Endianness is assumed to be little endian
/// Ordering is assumed to be 'C'.
struct TensorInfo {
    /// The type of each element of the tensor
    Dtype dtype;
    /// The shape of the tensor
    std::vector<std::size_t> shape;
    /// The offsets to find the data within the byte-buffer array.
    std::pair<std::size_t, std::size_t> data_offsets;
};

/// A view of a Tensor within the file.
/// Contains references to data within the full byte-buffer
/// And is thus a readable view of a single tensor
class TensorView {
   public:
    TensorView(Dtype dtype, std::vector<std::size_t> shape, std::span<const std::byte> data)
        : dtype_(dtype), shape_(std::move(shape)), data_(data) {}

    [[nodiscard]] Dtype dtype() const noexcept { return dtype_; }
    [[nodiscard]] const std::vector<std::size_t>& shape() const noexcept { return shape_; }
    [[nodiscard]] std::span<const std::byte> data() const noexcept { return data_; }
    [[nodiscard]] std::size_t data_len() const noexcept { return data_.size_bytes(); }

   private:
    Dtype dtype_;
    std::vector<std::size_t> shape_;
    std::span<const std::byte> data_;
};

/// The stuct representing the header of safetensor files which allow
/// indexing into the raw byte-buffer array and how to interpret it.
struct Metadata {
    std::optional<std::unordered_map<std::string, std::string>> metadata;
    std::vector<TensorInfo> tensors;
    std::unordered_map<std::string, std::size_t, string_transparent_hash, std::equal_to<>> index_map;

    /// Validates metadata and returns the total buffer size needed
    /// Throws SafeTensorException on validation failure
    std::size_t validate() const;
};

class SafeTensors {
   public:
    explicit SafeTensors(const std::filesystem::path& path);

    SafeTensors(const SafeTensors&) = delete;
    SafeTensors& operator=(const SafeTensors&) = delete;
    SafeTensors(SafeTensors&&) noexcept = default;
    SafeTensors& operator=(SafeTensors&&) noexcept = default;
    ~SafeTensors() = default;

    /// Given a byte-buffer representing the whole safetensor file
    /// parses the header, and returns the size of the header + the parsed data.
    static std::pair<std::size_t, Metadata> read_metadata(std::span<const char> buffer);

    /// Allow the user to get a specific tensor within the SafeTensors.
    /// The tensor returned is merely a view and the data is not owned by this structure.
    TensorView tensor(std::string_view tensor_name) const;

    /// Returns the tensors contained within the SafeTensors.
    /// The tensors returned are merely views and the data is not owned by this structure.
    std::vector<std::pair<std::string_view, TensorView>> tensors() const;

    /// Return the names of the tensors within the SafeTensors.
    /// These are used as keys to access to the actual tensors.
    std::vector<std::string_view> names() const;

    /// Return how many tensors are currently stored within the SafeTensors.
    [[nodiscard]] std::size_t size() const noexcept;

    /// Indicate if the SafeTensors contains or not any tensor.
    [[nodiscard]] bool empty() const noexcept;

   private:
    mio::mmap_source mmap_;
    Metadata metadata_;
    const std::byte* tensor_data_;
};

}  // namespace safetensors
