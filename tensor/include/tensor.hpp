#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

#include "safetensors.hpp"

namespace infer {

constexpr std::size_t TENSOR_ALIGNMENT = 64;

template <typename T>
class Tensor {
   public:
    using value_type = T;
    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;

   private:
    struct AlignedDeleter {
        void operator()(T* ptr) const { std::free(ptr); }  // NOLINT
    };

    std::unique_ptr<T[], AlignedDeleter> data_;  // NOLINT
    shape_type shape_;
    size_type size_ = 0;
    bool owns_data_ = true;
    T* view_ptr_ = nullptr;

    void allocate() {
        if (size_ == 0) return;
        void* ptr = std::aligned_alloc(TENSOR_ALIGNMENT, size_ * sizeof(T));  // NOLINT
        if (ptr == nullptr) throw std::bad_alloc();
        data_.reset(static_cast<T*>(ptr));
    }

    void compute_size() {
        size_ = 1;
        for (auto dim : shape_) size_ *= dim;
    }

    template <typename... Indices>
    [[nodiscard]] size_type get_flat_index(Indices... indices) const {
        static_assert((std::is_integral_v<Indices> && ...), "Indices must be integral types");
        constexpr size_type num_indices = sizeof...(Indices);

        if (num_indices != shape_.size()) {
            throw std::invalid_argument("Number of indices does not match tensor rank");
        }

        size_type flat_idx = 0;
        size_type stride = 1;
        size_type idx_array[num_indices] = {static_cast<size_type>(indices)...};

        for (size_type i = shape_.size(); i > 0; --i) {
            size_type idx = idx_array[i - 1];
            flat_idx += idx * stride;
            stride *= shape_[i - 1];
        }

        return flat_idx;
    }

   public:
    Tensor() = default;

    explicit Tensor(shape_type shape) : shape_(std::move(shape)) {
        compute_size();
        allocate();
    }

    Tensor(T* data, shape_type shape) : shape_(std::move(shape)), owns_data_(false), view_ptr_(data) { compute_size(); }

    explicit Tensor(const safetensors::TensorView& view) : shape_(view.shape()), owns_data_(false) {
        if constexpr (std::is_same_v<T, float>) {
            if (view.dtype() != safetensors::Dtype::F32) {
                throw std::runtime_error("Dtype mismatch: expected F32");
            }
        } else if constexpr (std::is_same_v<T, double>) {
            if (view.dtype() != safetensors::Dtype::F64) {
                throw std::runtime_error("Dtype mismatch: expected F64");
            }
        }
        compute_size();
        view_ptr_ = const_cast<T*>(reinterpret_cast<const T*>(view.data().data()));
    }

    [[nodiscard]] T* data() noexcept { return owns_data_ ? data_.get() : view_ptr_; }
    [[nodiscard]] const T* data() const noexcept { return owns_data_ ? data_.get() : view_ptr_; }

    [[nodiscard]] const shape_type& shape() const noexcept { return shape_; }
    [[nodiscard]] size_type size() const noexcept { return size_; }
    [[nodiscard]] size_type rank() const noexcept { return shape_.size(); }

    [[nodiscard]] T& operator[](size_type idx) { return data()[idx]; }
    [[nodiscard]] const T& operator[](size_type idx) const { return data()[idx]; }

    void fill(T value) { std::fill_n(data(), size_, value); }
    void zeros() { fill(T(0)); }
    void ones() { fill(T(1)); }

    [[nodiscard]] Tensor reshape(shape_type new_shape) const {
        size_type new_size = 1;
        for (auto dim : new_shape) new_size *= dim;
        if (new_size != size_) throw std::runtime_error("Cannot reshape: size mismatch");
        return {const_cast<T*>(data()), std::move(new_shape)};
    }

    [[nodiscard]] Tensor clone() const {
        Tensor copy(shape_);
        std::copy_n(data(), size_, copy.data());
        return copy;
    }
};

}  // namespace infer
