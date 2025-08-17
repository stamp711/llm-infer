#pragma once

#include <cstddef>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "shape/shape_concepts.hpp"

namespace tensor {

template <typename IndexT = std::size_t>
class DynamicShape {
   public:
    DynamicShape(std::initializer_list<IndexT> dims)
        : dims_(dims), size_(std::accumulate(dims_.begin(), dims_.end(), IndexT(1), std::multiplies<IndexT>())) {}
    DynamicShape(std::vector<IndexT> dims)
        : dims_(std::move(dims)),
          size_(std::accumulate(dims_.begin(), dims_.end(), IndexT(1), std::multiplies<IndexT>())) {}

    using index_type = IndexT;
    [[nodiscard]] constexpr index_type rank() const noexcept { return dims_.size(); }
    [[nodiscard]] constexpr index_type dim(index_type i) const { return dims_[i]; }
    [[nodiscard]] constexpr index_type size() const noexcept { return size_; }

    template <bool BoundsCheck = false>
    [[nodiscard]] index_type get_flat_index(std::span<const IndexT> idx) const {
        if constexpr (BoundsCheck) {
            if (idx.size() != rank()) {
                throw std::invalid_argument("Invalid number of indices");
            }
            for (auto i = 0; i < rank(); ++i) {
                if (idx[i] < 0 || idx[i] >= dims_[i]) {
                    throw std::out_of_range("Index out of bounds");
                }
            }
        }
        index_type flat = 0;
        index_type stride = 1;
        for (auto d = rank(); d > 0; --d) {
            flat += idx[d - 1] * stride;
            stride *= dims_[d - 1];
        }
        return flat;
    }

   private:
    std::vector<IndexT> dims_;
    index_type size_;
};
static_assert(ShapePolicy<DynamicShape<std::size_t>>, "DynamicShape does not satisfy ShapePolicy");

// template <typename IndexT, std::size_t Rank>
// class FixedRankShape {
//    public:
//     FixedRankShape(std::array<IndexT, Rank> dims)
//         : dims_(dims), size_(std::accumulate(dims_.begin(), dims_.end(), IndexT(1), std::multiplies<IndexT>())) {}

//     using index_type = IndexT;
//     [[nodiscard]] static constexpr index_type rank() noexcept { return Rank; }
//     [[nodiscard]] constexpr index_type size() const noexcept { return size_; }

//     template <index_type i>
//         requires(i >= 0 && i < Rank)
//     [[nodiscard]] consteval index_type dim() {
//         return dims_[i];
//     }

//     [[nodiscard]] constexpr index_type dim(index_type i) const { return dims_[i]; }

//     [[nodiscard]] index_type flatten(std::span<const IndexT, Rank> idx) const {
//         index_type flat = 0;
//         index_type stride = 1;
//         for (auto d = Rank - 1; d >= 0; --d) {
//             flat += idx[d] * stride;
//             stride *= dims_[d];
//         }
//         return flat;
//     }

//     template <typename... Idxs>
//         requires(sizeof...(Idxs) == Rank)
//     [[nodiscard]] constexpr index_type flatten(Idxs... idx) const noexcept {
//         std::array<index_type, Rank> indices = {static_cast<index_type>(idx)...};
//         std::span<const index_type> span(indices.data(), indices.size());
//         return flatten(span);
//     }

//    private:
//     index_type size_;
//     std::array<IndexT, Rank> dims_;
// };
// static_assert(ShapePolicy<FixedRankShape<std::size_t, 3>>, "FixedRankShape does not satisfy ShapePolicy");

// template <typename IndexT, IndexT... Dims>
// class StaticShape {
//    public:
//     using index_type = IndexT;
//     [[nodiscard]] static consteval index_type rank() noexcept { return sizeof...(Dims); }
//     [[nodiscard]] static consteval index_type size() noexcept { return (Dims * ...); }

//     template <index_type i>
//         requires(i >= 0 && i < sizeof...(Dims))
//     [[nodiscard]] static consteval index_type dim() {
//         return dims_[i];
//     }

//     template <bool BoundsCheck = false>
//     [[nodiscard]] static consteval index_type dim(index_type i) {
//         if constexpr (BoundsCheck) {
//             static_assert(i >= 0 && i < rank(), "Index out of bounds");
//         }
//         return dims_[i];
//     }

//     [[nodiscard]] constexpr index_type get_flat_index(std::span<const IndexT, sizeof...(Dims)> idx) const {
//         index_type flat = 0;
//         index_type stride = 1;
//         for (std::size_t d = sizeof...(Dims); d > 0; --d) {
//             flat += idx[d - 1] * stride;
//             stride *= dims_[d - 1];
//         }
//         return flat;
//     }

//     template <typename... Idxs>
//         requires(sizeof...(Idxs) == sizeof...(Dims))
//     [[nodiscard]] consteval index_type get_flat_index(Idxs... idx) const noexcept {
//         std::array<index_type, sizeof...(Dims)> indices = {static_cast<index_type>(idx)...};
//         std::span<const index_type, sizeof...(Dims)> span(indices.data(), indices.size());
//         return get_flat_index(span);
//     }

//    private:
//     static constexpr std::array<index_type, sizeof...(Dims)> dims_ = {Dims...};
// };
// static_assert(ShapePolicy<StaticShape<std::size_t, 2, 3, 4>>, "StaticShape does not satisfy ShapePolicy");
// static_assert([] {
//     StaticShape<std::size_t, 2, 2> s;
//     return s.get_flat_index(1, 1) == 3;  // must be constexpr-evaluated
// }());

}  // namespace tensor
