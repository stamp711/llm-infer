#pragma once

#include <concepts>
#include <cstddef>
#include <type_traits>

namespace infer {

// Base template for tensor type traits - specialized by each tensor class
template <typename D>
struct TensorInnerTypes;

// Helper to compute product of compile-time values
template <std::size_t... Values>
struct product {
    static constexpr std::size_t value = (Values * ... * 1);
};

// Concept that documents and enforces requirements for TensorInnerTypes specializations
template<typename T>
concept TensorTraits = requires {
    // Required type aliases
    typename T::value_type;
    typename T::size_type;
    typename T::shape_type;

    // Required compile-time boolean flags
    { T::is_owning } -> std::convertible_to<bool>;
    { T::is_mutable } -> std::convertible_to<bool>;
    { T::is_static_shape } -> std::convertible_to<bool>;
    { T::is_static_rank } -> std::convertible_to<bool>;

    // size_type must be an integral type
    requires std::is_integral_v<typename T::size_type>;
} &&
// Conditional requirements for static-rank tensors
(T::is_static_rank ? requires {
    { T::rank } -> std::convertible_to<typename T::size_type>;
} : true) &&
// Conditional requirements for static-shape tensors
(T::is_static_shape ? requires {
    { T::static_size } -> std::convertible_to<typename T::size_type>;
    { T::static_shape } -> std::convertible_to<typename T::shape_type>;
    // Static tensors also require storage_type
    typename T::storage_type;
} : true);

}  // namespace infer
