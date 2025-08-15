#include "safetensors.hpp"

#include <algorithm>
#include <cstring>
#include <nlohmann/json.hpp>
#include <string_view>

namespace safetensors {

// Constants from Rust implementation
constexpr std::size_t MAX_HEADER_SIZE = 100'000'000;
constexpr std::size_t N_LEN = sizeof(std::uint64_t);

// Helper function to parse Dtype from string
Dtype parse_dtype(const std::string& dtype_str) {
    if (dtype_str == "BOOL") return Dtype::BOOL;
    if (dtype_str == "F4") return Dtype::F4;
    if (dtype_str == "F6_E2M3") return Dtype::F6_E2M3;
    if (dtype_str == "F6_E3M2") return Dtype::F6_E3M2;
    if (dtype_str == "U8") return Dtype::U8;
    if (dtype_str == "I8") return Dtype::I8;
    if (dtype_str == "F8_E5M2") return Dtype::F8_E5M2;
    if (dtype_str == "F8_E4M3") return Dtype::F8_E4M3;
    if (dtype_str == "F8_E8M0") return Dtype::F8_E8M0;
    if (dtype_str == "I16") return Dtype::I16;
    if (dtype_str == "U16") return Dtype::U16;
    if (dtype_str == "F16") return Dtype::F16;
    if (dtype_str == "BF16") return Dtype::BF16;
    if (dtype_str == "I32") return Dtype::I32;
    if (dtype_str == "U32") return Dtype::U32;
    if (dtype_str == "F32") return Dtype::F32;
    if (dtype_str == "F64") return Dtype::F64;
    if (dtype_str == "I64") return Dtype::I64;
    if (dtype_str == "U64") return Dtype::U64;

    throw SafeTensorException("Unknown dtype: " + dtype_str);
}

// Helper function to parse JSON into Metadata
Metadata parse_json_metadata(const nlohmann::json& json_header) {
    Metadata metadata;

    // Parse metadata
    if (json_header.contains("__metadata__")) {
        auto meta_obj = json_header["__metadata__"];
        if (meta_obj.is_object()) {
            metadata.metadata = std::unordered_map<std::string, std::string>();
            for (const auto& [key, value] : meta_obj.items()) {
                if (value.is_string()) {
                    metadata.metadata->emplace(key, value.get<std::string>());
                }
            }
        }
    }

    // Parse tensors
    std::vector<std::pair<std::string, TensorInfo>> tensor_pairs;
    for (const auto& [tensor_name, tensor_data] : json_header.items()) {
        if (tensor_name == "__metadata__") continue;

        if (!tensor_data.is_object()) {
            throw SafeTensorException("Invalid tensor data for: " + tensor_name);
        }

        TensorInfo info;

        // Parse dtype
        if (!tensor_data.contains("dtype") || !tensor_data["dtype"].is_string()) {
            throw SafeTensorException("Missing or invalid dtype for tensor: " + tensor_name);
        }
        info.dtype = parse_dtype(tensor_data["dtype"].get<std::string>());

        // Parse shape
        if (!tensor_data.contains("shape") || !tensor_data["shape"].is_array()) {
            throw SafeTensorException("Missing or invalid shape for tensor: " + tensor_name);
        }
        for (const auto& dim : tensor_data["shape"]) {
            if (!dim.is_number_unsigned()) {
                throw SafeTensorException("Invalid shape dimension for tensor: " + tensor_name);
            }
            info.shape.push_back(dim.get<std::size_t>());
        }

        // Parse data_offsets
        if (!tensor_data.contains("data_offsets") || !tensor_data["data_offsets"].is_array() ||
            tensor_data["data_offsets"].size() != 2) {
            throw SafeTensorException("Missing or invalid data_offsets for tensor: " + tensor_name);
        }
        info.data_offsets.first = tensor_data["data_offsets"][0].get<std::size_t>();
        info.data_offsets.second = tensor_data["data_offsets"][1].get<std::size_t>();

        tensor_pairs.emplace_back(tensor_name, info);
    }

    // Sort tensors by offset to match Rust implementation
    std::ranges::sort(tensor_pairs,
                      [](const auto& a, const auto& b) { return a.second.data_offsets < b.second.data_offsets; });

    // Build index map and tensor vector
    for (std::size_t i = 0; i < tensor_pairs.size(); ++i) {
        metadata.index_map.emplace(tensor_pairs[i].first, i);
        metadata.tensors.push_back(tensor_pairs[i].second);
    }

    return metadata;
}

std::size_t Metadata::validate() const {
    std::size_t start = 0;
    for (std::size_t i = 0; i < tensors.size(); ++i) {
        const auto& info = tensors[i];
        const auto [s, e] = info.data_offsets;

        if (s != start || e < s) {
            // Find tensor name for error reporting
            std::string tensor_name = "no_tensor";
            for (const auto& [name, index] : index_map) {
                if (index == i) {
                    tensor_name = name;
                    break;
                }
            }
            throw SafeTensorException("Invalid offset for tensor '" + tensor_name + "'");
        }

        start = e;

        // Calculate number of elements
        std::size_t nelements = 1;
        for (std::size_t dim : info.shape) {
            if (nelements > std::numeric_limits<std::size_t>::max() / dim) {  // Check for overflow
                throw SafeTensorException("Validation overflow computing number of elements");
            }
            nelements *= dim;
        }

        // Calculate total bits needed
        if (nelements > std::numeric_limits<std::size_t>::max() / dtype_bitsize(info.dtype)) {  // Check for overflow
            throw SafeTensorException("Validation overflow computing total bits needed");
        }
        std::size_t nbits = nelements * dtype_bitsize(info.dtype);
        if (nbits % 8 != 0) {
            throw SafeTensorException("Misaligned slice - tensor size not byte-aligned");
        }

        std::size_t size = nbits / 8;
        if (e - s != size) {
            throw SafeTensorException("Invalid tensor info - size mismatch");
        }
    }
    return start;
}

std::pair<std::size_t, Metadata> SafeTensors::read_metadata(std::span<const char> buffer) {
    if (buffer.size() < N_LEN) {
        throw SafeTensorException("Header too small");
    }

    // Read header size (8 bytes, little-endian)
    std::uint64_t n_u64 = 0;
    std::memcpy(&n_u64, buffer.data(), N_LEN);
    if (n_u64 > MAX_HEADER_SIZE) {
        throw SafeTensorException("Header too large");
    }
    auto n = static_cast<std::size_t>(n_u64);

    auto stop = n + N_LEN;
    if (stop > buffer.size()) {
        throw SafeTensorException("Invalid header length");
    }

    // Extract and parse JSON header
    std::string header_str(buffer.data() + N_LEN, n);

    // Parse JSON
    nlohmann::json json_header;
    try {
        json_header = nlohmann::json::parse(header_str);
    } catch (const nlohmann::json::exception& e) {
        throw SafeTensorException("Invalid JSON in header: " + std::string(e.what()));
    }

    // Parse metadata from JSON
    Metadata metadata = parse_json_metadata(json_header);

    // Validate metadata
    auto buffer_end = metadata.validate();
    if (buffer_end + N_LEN + n != buffer.size()) {
        throw SafeTensorException("Metadata incomplete buffer");
    }

    return {n, std::move(metadata)};
}

SafeTensors::SafeTensors(const std::filesystem::path& path) : mmap_(path.string()) {
    std::span<const char> buffer(mmap_.data(), mmap_.size());
    auto [n, metadata] = read_metadata(buffer);
    metadata_ = std::move(metadata);

    // Calculate tensor data start position
    tensor_data_ = reinterpret_cast<const std::byte*>(mmap_.data() + N_LEN + n);
}

TensorView SafeTensors::tensor(std::string_view tensor_name) const {
    auto it = metadata_.index_map.find(tensor_name);
    if (it == metadata_.index_map.end()) {
        throw SafeTensorException(std::string("Tensor not found: ") + std::string(tensor_name));
    }

    const auto& info = metadata_.tensors[it->second];
    const auto [start, end] = info.data_offsets;

    std::span<const std::byte> data(tensor_data_ + start, tensor_data_ + end);
    return {info.dtype, info.shape, data};
}

std::vector<std::pair<std::string_view, TensorView>> SafeTensors::tensors() const {
    std::vector<std::pair<std::string_view, TensorView>> result;
    result.reserve(metadata_.index_map.size());

    for (const auto& [name, index] : metadata_.index_map) {
        const auto& info = metadata_.tensors[index];
        const auto [start, end] = info.data_offsets;

        std::span<const std::byte> data(tensor_data_ + start, tensor_data_ + end);
        result.emplace_back(name, TensorView{info.dtype, info.shape, data});
    }

    return result;
}

std::vector<std::string_view> SafeTensors::names() const {
    std::vector<std::string_view> result;
    result.reserve(metadata_.index_map.size());

    for (const auto& [name, _] : metadata_.index_map) {
        result.emplace_back(name);
    }

    return result;
}

std::size_t SafeTensors::size() const noexcept { return metadata_.tensors.size(); }

bool SafeTensors::empty() const noexcept { return metadata_.tensors.empty(); }

}  // namespace safetensors
