#pragma once

#include <array>
#include <utility>

#include "tensor_base.hpp"

namespace infer {

// Forward declaration
template <typename T, std::size_t... Dims>
class TensorStaticView;

// Traits specialization for TensorStaticView
template <typename T, std::size_t... Dims>
struct TensorInnerTypes<TensorStaticView<T, Dims...>> {
    using value_type = T;
    using size_type = std::size_t;
    using shape_type = std::array<size_type, sizeof...(Dims)>;
    static constexpr bool is_owning = false;
    static constexpr bool is_mutable = false;
    static constexpr bool is_static_shape = true;
    static constexpr bool is_static_rank = true;
    static constexpr size_type rank = sizeof...(Dims);
    static constexpr size_type static_size = product<Dims...>::value;
    static constexpr shape_type static_shape = {Dims...};
};

// Read-only view of fully static tensor - doesn't own data
template <typename T, std::size_t... Dims>
class TensorStaticView : public ReadOnlyTensorBase<TensorStaticView<T, Dims...>> {
   public:
    using base_type = ReadOnlyTensorBase<TensorStaticView<T, Dims...>>;
    using inner_types = TensorInnerTypes<TensorStaticView<T, Dims...>>;
    using typename base_type::shape_type;
    using typename base_type::size_type;
    using typename base_type::value_type;

    static constexpr size_type static_size = inner_types::static_size;
    static constexpr size_type static_rank = sizeof...(Dims);

   private:
    const T* data_ptr_ = nullptr;  // Non-owning pointer to const data
    static constexpr shape_type shape_ = {Dims...};

   public:
    TensorStaticView() = default;

    // View from raw const pointer
    explicit TensorStaticView(const T* data) : data_ptr_(data) {}

    // CRTP implementations
    [[nodiscard]] const T* data_impl() const noexcept { return data_ptr_; }
    [[nodiscard]] const shape_type& shape_impl() const noexcept { return shape_; }
    [[nodiscard]] constexpr size_type size_impl() const noexcept { return static_size; }
    [[nodiscard]] constexpr size_type rank_impl() const noexcept { return static_rank; }

    // Compile-time bounds checked read-only indexing
    template <typename... Indices>
    [[nodiscard]] const value_type& operator()(Indices... indices) const {
        static_assert(sizeof...(indices) == static_rank, "Number of indices must match tensor rank");
        return data_ptr_[get_flat_index(indices...)];
    }

   private:
    // Compile-time optimized index calculation
    template <typename... Indices>
    [[nodiscard]] static constexpr size_type get_flat_index(Indices... indices) {
        static_assert((std::is_integral_v<Indices> && ...), "Indices must be integral types");

        size_type flat_idx = 0;
        size_type stride = 1;
        std::array<size_type, static_rank> idx_array = {static_cast<size_type>(indices)...};
        std::array<size_type, static_rank> dims_array = {Dims...};

        // Loop can be unrolled at compile time
        for (size_type i = static_rank; i > 0; --i) {
            flat_idx += idx_array[i - 1] * stride;
            stride *= dims_array[i - 1];
        }

        return flat_idx;
    }
};

}  // namespace infer
