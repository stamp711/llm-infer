#pragma once

#include <stdexcept>
#include <utility>

#include "shape/shape_concepts.hpp"
#include "storage/storage_concepts.hpp"

namespace tensor {

// The single unified tensor class
template <typename T, ShapePolicy Shape, TensorStorage Storage>
    requires std::same_as<typename Storage::value_type, T> &&
             std::same_as<typename Storage::index_type, typename Shape::index_type>
class Tensor {
   private:
    // Storage and shape management
    [[no_unique_address]] Storage storage_;
    [[no_unique_address]] Shape shape_;

   public:
    using index_type = Shape::index_type;
    static constexpr bool is_valid = true;

    // Constructor taking shape and storage
    template <bool VerifySize = true>
    Tensor(Shape shape, Storage storage) : storage_(std::move(storage)), shape_(std::move(shape)) {
        if constexpr (VerifySize) {
            if (storage_.size() != shape_.size()) {
                throw std::invalid_argument("Storage size must match shape size");
            }
        }
    }

    [[nodiscard]] constexpr index_type size() const noexcept { return shape_.size(); }
    [[nodiscard]] constexpr std::size_t rank() const noexcept { return shape_.rank(); }

    // Data access methods
    [[nodiscard]] T* data() noexcept
        requires MutableTensorStorage<Storage>
    {
        return storage_.data();
    }

    [[nodiscard]] const T* data() const noexcept { return storage_.data(); }

    // Indexing operators
    template <bool BoundsCheck = false>
    [[nodiscard]] const T& operator[](index_type idx) const {
        if constexpr (BoundsCheck) {
            if (idx < 0 || idx >= size()) {
                throw std::out_of_range("Index out of range");
            }
        }
        return data()[idx];
    }

    [[nodiscard]] T& operator[](index_type idx)
        requires MutableTensorStorage<Storage>
    {
        return data()[idx];
    }

    // // Ensure storage matches computed size
    // void ensure_storage_size() {
    //     if constexpr (ResizableTensorStorage<Storage, value_type>) {
    //         storage_.resize(size_);
    //     } else {
    //         if (storage_.size() != size_) {
    //             throw std::invalid_argument("Storage size must match tensor size");
    //         }
    //     }
    // }

    // public:
    // Default constructor
    // Tensor() {
    //     if constexpr (is_static_shape) {
    //         shape_ = static_shape_;
    //         size_ = static_size;
    //         if constexpr (ResizableTensorStorage<Storage, T>) {
    //             storage_.resize(size_);
    //         }
    //     }
    // }

    // Constructor with storage and shape (for all types)
    // Tensor(Storage storage, shape_type shape) : storage_(std::move(storage)), shape_(std::move(shape)) {
    //     compute_size();
    //     if (storage_.size() != size_) {
    //         throw std::invalid_argument("Storage size must match shape size");
    //     }
    // }

    // Shape-only constructor for resizable storage
    // explicit Tensor(shape_type shape)
    //     requires ResizableTensorStorage<Storage, T> && (!is_static_shape)
    //     : shape_(std::move(shape)) {
    //     compute_size();
    //     ensure_storage_size();
    // }

    // Initializer list constructor for dynamic/fixed rank
    // Tensor(std::initializer_list<dim_type> shape)
    //     requires ResizableTensorStorage<Storage, T> && (!is_static_shape)
    // {
    //     if constexpr (is_static_rank) {
    //         if (shape.size() != static_rank) {
    //             throw std::invalid_argument("Shape size must match tensor rank");
    //         }
    //         std::copy(shape.begin(), shape.end(), shape_.begin());
    //     } else {
    //         shape_ = shape_type(shape);
    //     }
    //     compute_size();
    //     ensure_storage_size();
    // }

    // Storage-only constructor for static shapes
    // explicit constexpr Tensor(Storage storage)
    //     requires is_static_shape
    //     : storage_(std::move(storage)), shape_(static_shape_) {
    //     if (storage_.size() != static_size_) {
    //         throw std::invalid_argument("Storage size must match static size");
    //     }
    // }

    // Fill constructor
    // explicit Tensor(value_type fill_value)
    //     requires MutableTensorStorage<Storage, T> && ResizableTensorStorage<Storage, T>
    //     : Tensor() {
    //     fill(fill_value);
    // }

    // Core interface methods
    // [[nodiscard]] const shape_type& shape() const noexcept {
    //     if constexpr (is_static_shape) {
    //         return static_shape_;
    //     } else {
    //         return shape_;
    //     }
    // }
};

}  // namespace tensor
