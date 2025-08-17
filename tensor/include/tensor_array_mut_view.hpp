#pragma once

#include <utility>
#include <vector>

#include "tensor_base.hpp"

namespace infer {

// Forward declaration
template <typename T>
class TensorArrayMutView;

// Traits specialization for TensorArrayMutView
template <typename T>
struct TensorInnerTypes<TensorArrayMutView<T>> {
    using value_type = T;
    using size_type = std::size_t;
    using shape_type = std::vector<size_type>;
    static constexpr bool is_owning = false;
    static constexpr bool is_mutable = true;
    static constexpr bool is_static_shape = false;
    static constexpr bool is_static_rank = false;
};

// Mutable view of dynamic tensor - doesn't own data
template <typename T>
class TensorArrayMutView : public MutableViewStridedTensorBase<TensorArrayMutView<T>> {
   public:
    using base_type = MutableViewStridedTensorBase<TensorArrayMutView<T>>;
    using base_type::compute_size;
    using base_type::data_ptr_;
    using base_type::shape_;
    using base_type::size_;
    using typename base_type::shape_type;
    using typename base_type::size_type;
    using typename base_type::value_type;

    TensorArrayMutView() = default;

    // View from raw pointer
    TensorArrayMutView(T* data, shape_type shape) : base_type() {
        data_ptr_ = data;
        shape_ = std::move(shape);
        compute_size();
    }

    [[nodiscard]] size_type rank_impl() const noexcept { return shape_.size(); }

    // Variadic indexing with runtime bounds checking
    template <typename... Indices>
    [[nodiscard]] value_type& operator()(Indices... indices) {
        return data_ptr_[get_flat_index(indices...)];
    }

    template <typename... Indices>
    [[nodiscard]] const value_type& operator()(Indices... indices) const {
        return data_ptr_[get_flat_index(indices...)];
    }

   private:
    template <typename... Indices>
    [[nodiscard]] size_type get_flat_index(Indices... indices) const {
        static_assert((std::is_integral_v<Indices> && ...), "Indices must be integral types");
        constexpr size_type num_indices = sizeof...(Indices);

        if (num_indices != shape_.size()) {
            throw std::invalid_argument("Number of indices does not match tensor rank");
        }

        size_type flat_idx = 0;
        size_type stride = 1;
        std::array<size_type, num_indices> idx_array = {static_cast<size_type>(indices)...};

        for (size_type i = shape_.size(); i > 0; --i) {
            size_type idx = idx_array[i - 1];
            flat_idx += idx * stride;
            stride *= shape_[i - 1];
        }

        return flat_idx;
    }
};

}  // namespace infer
