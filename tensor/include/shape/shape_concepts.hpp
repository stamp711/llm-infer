#pragma once

#include <concepts>
#include <cstddef>

namespace tensor {

// Helper concept for shape container requirements
template <typename ShapeType>
concept ShapeContainer = requires {
    requires std::integral<typename ShapeType::value_type>;

    // Container interface
    { std::declval<ShapeType>().size() } -> std::same_as<std::size_t>;
    { std::declval<ShapeType>()[0] } -> std::convertible_to<typename ShapeType::value_type>;
    { std::declval<ShapeType>().begin() };
    { std::declval<ShapeType>().end() };

    // Element type consistency and value semantics
    // requires std::is_same_v<typename ShapeType::value_type, DimType>;
    requires std::copyable<ShapeType>;
};

// // Concept for shape policies used by Tensor
// template <typename Policy>
// concept ShapePolicy = requires {
//     // Container requirements for shape_type
//     requires ShapeContainer<typename Policy::shape_type>;

//     // Required compile-time flags
//     { Policy::is_static_shape } -> std::convertible_to<bool>;
//     { Policy::is_static_rank } -> std::convertible_to<bool>;

//     // Conditional requirements for static rank policies
//     requires(!Policy::is_static_rank) || requires {
//         { Policy::rank } -> std::same_as<std::size_t>;
//     };

//     // Can only be static shape if rank is static
//     requires(Policy::is_static_rank || !Policy::is_static_shape);

//     // Conditional requirements for static shape policies
//     requires(!Policy::is_static_shape) || requires {
//         {
//             Policy::static_size
//         }
//         -> std::same_as<typename Policy::shape_type::value_type>;  // size is the product of dims, so same type as
//         dims Policy::static_shape;
//     };
// };

template <typename Shape>
concept ShapePolicy = requires(const Shape s, size_t i) {
    requires std::integral<typename Shape::index_type>;

    { s.rank() } noexcept -> std::convertible_to<typename Shape::index_type>;
    { s.size() } noexcept -> std::convertible_to<typename Shape::index_type>;
    { s.dim(i) } -> std::convertible_to<typename Shape::index_type>;

    // {
    //     s.get_flat_index(std::declval<std::span<const typename Shape::index_type>>())
    // } -> std::convertible_to<typename Shape::index_type>;
};

}  // namespace tensor
