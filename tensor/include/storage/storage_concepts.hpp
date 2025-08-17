#pragma once

#include <concepts>

namespace tensor {

// Core storage concept - defines the minimal interface for tensor storage
template <typename S>
concept TensorStorage = requires(const S& storage) {
    // Type definitions
    typename S::value_type;
    typename S::index_type;

    // Basic interface
    { storage.data() } noexcept -> std::same_as<const typename S::value_type*>;
    { storage.size() } -> std::same_as<typename S::index_type>;
    { storage.empty() } -> std::same_as<bool>;
};
// && std::copyable<S>;

// Mutable storage concept - adds mutable data access
template <typename S>
concept MutableTensorStorage = TensorStorage<S> && requires(S& storage) {
    { storage.data() } noexcept -> std::convertible_to<typename S::value_type*>;
};

// Resizable storage concept - adds dynamic sizing capability
template <typename S>
concept ResizableTensorStorage = MutableTensorStorage<S> && requires(S& storage, typename S::index_type size) {
    storage.resize(size);
    { storage.capacity() } -> std::convertible_to<typename S::index_type>;
};

// Cloneable storage concept - can create optimized deep copies
template <typename S>
concept CloneableTensorStorage = TensorStorage<S> && requires(const S& storage) {
    { storage.clone() } -> std::same_as<S>;
};

}  // namespace tensor
