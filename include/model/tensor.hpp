#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

#include "gguf.hpp"
#include "model/config.hpp"
#include "model/device.hpp"

// Helper methods for tensor loading
[[nodiscard]]
inline const TensorInfo& get_tensor_required(const GGUF& gguf, std::string_view name) {
    const TensorInfo* tensor = gguf.get_tensor(std::string(name));
    if (tensor == nullptr) {
        throw std::runtime_error(std::string("Required tensor not found: ") + std::string(name));
    }
    return *tensor;
}

enum class Ownership : std::uint8_t { Reference, CPUOwned, CUDAOwned };

template <typename T>
class Tensor {
   public:
    ~Tensor() {
        switch (ownership_) {
            case Ownership::Reference: break;
            case Ownership::CPUOwned:
                std::free(reinterpret_cast<void*>(const_cast<std::remove_const_t<T>*>(data_)));  // NOLINT
                data_ = nullptr;
                break;
            case Ownership::CUDAOwned:
                CUDAContext::free(reinterpret_cast<void*>(const_cast<std::remove_const_t<T>*>(data_)));
                data_ = nullptr;
                break;
        }
    }

    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    Tensor(Tensor&& other) noexcept : Tensor() { swap(*this, other); }
    Tensor& operator=(Tensor&& other) noexcept {
        if (this != &other) {
            swap(*this, other);
        }
        return *this;
    }

    friend void swap(Tensor& lhs, Tensor& rhs) noexcept {
        std::swap(lhs.data_, rhs.data_);
        std::swap(lhs.quantization_, rhs.quantization_);
        std::swap(lhs.ownership_, rhs.ownership_);
    }

    Tensor() noexcept : data_(nullptr), quantization_(QuantizationType::FP32), ownership_(Ownership::Reference) {}

    explicit Tensor(T* data, QuantizationType quantization, Ownership ownership) noexcept
        : data_(data), quantization_(quantization), ownership_(ownership) {}

    static Tensor allocate(QuantizationType quantization, std::size_t length, DeviceType device_type) {
        auto qsize = quantization_size(quantization);
        if (qsize != sizeof(T)) throw std::invalid_argument("Quantization size mismatch");
        auto size = qsize * length;
        switch (device_type) {
            case DeviceType::CPU_UnAligned: {
                auto* ptr = static_cast<T*>(std::malloc(size));  // NOLINT
                return Tensor(ptr, quantization, Ownership::CPUOwned);
            }
            case DeviceType::CPU: {
                auto* ptr = static_cast<T*>(std::aligned_alloc(CPU_ALIGN, size));  // NOLINT
                return Tensor(ptr, quantization, Ownership::CPUOwned);
            }
            case DeviceType::CUDA: {
                auto* ptr = static_cast<T*>(CUDAContext::malloc(size));
                return Tensor(ptr, quantization, Ownership::CUDAOwned);
            }
            default: throw std::invalid_argument("Invalid ownership type, must be owned");
        }
    }

    [[nodiscard]]
    bool has_value() const noexcept {
        return data_ != nullptr;
    }
    [[nodiscard]]
    QuantizationType quantization() const noexcept {
        return quantization_;
    }
    [[nodiscard]]
    T* data() const noexcept {
        return data_;
    }
    void set(T* data) { data_ = data; }

   private:
    T* data_ = nullptr;
    QuantizationType quantization_;
    Ownership ownership_;
};

template <typename T>
Tensor(T*, QuantizationType, DeviceType) -> Tensor<T>;

// Helper function to validate tensor shape and quantization
inline void check_tensor(const TensorInfo& tensor, const std::vector<uint32_t>& expected_shape,
                         QuantizationType expected_quantization) {
    if (!std::ranges::equal(tensor.dimensions, expected_shape)) {
        std::string actual = "[";
        for (size_t i = 0; i < tensor.dimensions.size(); ++i) {
            if (i > 0) actual += ", ";
            actual += std::to_string(tensor.dimensions[i]);
        }
        actual += "]";

        std::string expected = "[";
        for (size_t i = 0; i < expected_shape.size(); ++i) {
            if (i > 0) expected += ", ";
            expected += std::to_string(expected_shape[i]);
        }
        expected += "]";

        throw std::runtime_error("Tensor '" + tensor.name + "' shape mismatch: expected " + expected + ", got " +
                                 actual);
    }

    auto actual_quantization = quantization_from_gguf(tensor.type);
    if (actual_quantization != expected_quantization) {
        throw std::runtime_error("Tensor '" + tensor.name + "' quantization mismatch");
    }
}

// Specialization for const pointers from GGUF
template <typename T>
Tensor<const T> tensor_from_gguf(const TensorInfo& tensor, DeviceType device_type) {
    auto quantization = quantization_from_gguf(tensor.type);

    switch (device_type) {
        case DeviceType::CPU_UnAligned:
            return Tensor<const T>{reinterpret_cast<const T*>(tensor.data), quantization, Ownership::Reference};
        case DeviceType::CPU:
            check_cpu_alignment(tensor.data);
            return Tensor<const T>{reinterpret_cast<const T*>(tensor.data), quantization, Ownership::Reference};
        case DeviceType::CUDA: {
            std::size_t size = tensor.elements() * quantization_size(quantization);
            auto* device_data = reinterpret_cast<T*>(CUDAContext::copy_to_device_async(tensor.data, size));
            // We still return Tensor<const T> for consistency
            return Tensor<const T>{device_data, quantization, Ownership::CUDAOwned};
        }
        default: throw std::runtime_error("Invalid device type");
    }
}
