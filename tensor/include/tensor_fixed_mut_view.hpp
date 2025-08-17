#pragma once

#include <array>
#include <utility>

#include "tensor_base.hpp"

namespace infer {

// Forward declaration
template <typename T, std::size_t N>
class TensorFixedMutView;

// Traits specialization for TensorFixedMutView
template <typename T, std::size_t N>
struct TensorInnerTypes<TensorFixedMutView<T, N>> {
    using value_type = T;
    using size_type = std::size_t;
    using shape_type = std::array<size_type, N>;
    static constexpr bool is_owning = false;
    static constexpr bool is_mutable = true;
    static constexpr bool is_static_shape = false;
    static constexpr bool is_static_rank = true;
    static constexpr size_type rank = N;
};

// Mutable view of fixed-rank tensor - doesn't own data
template <typename T, std::size_t N>
class TensorFixedMutView : public MutableViewStridedTensorBase<TensorFixedMutView<T, N>> {
   public:
    using base_type = MutableViewStridedTensorBase<TensorFixedMutView<T, N>>;
    using base_type::compute_size;
    using base_type::data_ptr_;
    using base_type::shape_;
    using base_type::size_;
    using typename base_type::shape_type;
    using typename base_type::size_type;
    using typename base_type::value_type;

    TensorFixedMutView() = default;

    // View from raw pointer
    TensorFixedMutView(T* data, shape_type shape) : base_type() {
        data_ptr_ = data;
        shape_ = std::move(shape);
        compute_size();
    }

    // Initializer list constructor
    TensorFixedMutView(T* data, std::initializer_list<size_type> shape) : base_type() {
        if (shape.size() != N) {
            throw std::invalid_argument("Shape size does not match tensor rank");
        }
        data_ptr_ = data;
        std::copy(shape.begin(), shape.end(), shape_.begin());
        compute_size();
    }

    [[nodiscard]] constexpr size_type rank_impl() const noexcept { return N; }

    // Variadic indexing - no runtime rank check needed
    template <typename... Indices>
    [[nodiscard]] value_type& operator()(Indices... indices) {
        static_assert(sizeof...(indices) == N, "Number of indices must match tensor rank");
        return data_ptr_[get_flat_index(indices...)];
    }

    template <typename... Indices>
    [[nodiscard]] const value_type& operator()(Indices... indices) const {
        static_assert(sizeof...(indices) == N, "Number of indices must match tensor rank");
        return data_ptr_[get_flat_index(indices...)];
    }

   private:
    template <typename... Indices>
    [[nodiscard]] size_type get_flat_index(Indices... indices) const {
        static_assert((std::is_integral_v<Indices> && ...), "Indices must be integral types");

        size_type flat_idx = 0;
        size_type stride = 1;
        std::array<size_type, N> idx_array = {static_cast<size_type>(indices)...};

        // No runtime check needed - compile-time guarantees correct count
        for (size_type i = N; i > 0; --i) {
            size_type idx = idx_array[i - 1];
            flat_idx += idx * stride;
            stride *= shape_[i - 1];
        }

        return flat_idx;
    }
};

}  // namespace infer
