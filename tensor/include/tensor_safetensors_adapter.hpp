#pragma once

// SafeTensors adapter for tensor library - provides bridge functions
// This is the only file that depends on SafeTensors

#include <stdexcept>
#include <type_traits>

#include "safetensors.hpp"
#include "tensor_array.hpp"
#include "tensor_array_view.hpp"
#include "tensor_fixed_view.hpp"

namespace infer {

// Create read-only dynamic tensor view from SafeTensors
template <typename T>
[[nodiscard]] TensorArrayView<T> from_safetensor(const safetensors::TensorView& view) {
    if constexpr (std::is_same_v<T, float>) {
        if (view.dtype() != safetensors::Dtype::F32) {
            throw std::runtime_error("Dtype mismatch: expected F32");
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (view.dtype() != safetensors::Dtype::F64) {
            throw std::runtime_error("Dtype mismatch: expected F64");
        }
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    const auto* data_ptr = reinterpret_cast<const T*>(view.data().data());
    return TensorArrayView<T>(data_ptr, view.shape());
}

// Create read-only fixed-rank tensor view from SafeTensors
template <typename T, std::size_t N>
[[nodiscard]] TensorFixedView<T, N> from_safetensor_fixed(const safetensors::TensorView& view) {
    if constexpr (std::is_same_v<T, float>) {
        if (view.dtype() != safetensors::Dtype::F32) {
            throw std::runtime_error("Dtype mismatch: expected F32");
        }
    } else if constexpr (std::is_same_v<T, double>) {
        if (view.dtype() != safetensors::Dtype::F64) {
            throw std::runtime_error("Dtype mismatch: expected F64");
        }
    } else {
        throw std::runtime_error("Unsupported data type");
    }

    if (view.shape().size() != N) {
        throw std::runtime_error("Shape rank mismatch");
    }

    const auto* data_ptr = reinterpret_cast<const T*>(view.data().data());

    // Convert vector to array
    std::array<std::size_t, N> shape_array;
    std::copy(view.shape().begin(), view.shape().end(), shape_array.begin());

    return TensorFixedView<T, N>(data_ptr, shape_array);
}

// Copy data from SafeTensors into an owning dynamic tensor
template <typename T>
[[nodiscard]] TensorArray<T> copy_from_safetensor(const safetensors::TensorView& view) {
    // Create a view first to validate types and get the data
    auto tensor_view = from_safetensor<T>(view);

    // Create owning tensor and copy data
    TensorArray<T> tensor(tensor_view.shape());
    std::copy_n(tensor_view.data(), tensor_view.size(), tensor.data());

    return tensor;
}

}  // namespace infer
