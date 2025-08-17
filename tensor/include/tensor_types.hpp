#pragma once

#include "shape/shape_policies.hpp"
#include "storage/storage_types.hpp"
#include "tensor.hpp"

namespace tensor {

// ===== ALL TENSOR TYPES AS TYPE ALIASES =====

// Dynamic tensors (runtime rank and shape)
template <typename T>
using TensorDynamicHeap = Tensor<T, DynamicShape<>, HeapStorage<T>>;
static_assert(TensorDynamicHeap<float>::is_valid);

template <typename T>
using TensorDynamicView = Tensor<T, DynamicShape<>, ViewStorage<T>>;
static_assert(TensorDynamicView<float>::is_valid);

template <typename T>
using TensorDynamicMutView = Tensor<T, DynamicShape<>, MutViewStorage<T>>;
static_assert(TensorDynamicMutView<float>::is_valid);

// Fixed rank tensors (compile-time rank, runtime shape)
// template <typename T, std::size_t N>
// using TensorFixedHeap = Tensor<T, FixedRankShape<std::size_t, N>, HeapStorage<T>>;

// template <typename T, std::size_t N>
// using TensorFixedView = Tensor<T, FixedRankShape<std::size_t, N>, ViewStorage<T>>;

// template <typename T, std::size_t N>
// using TensorFixedMutView = Tensor<T, FixedRankShape<std::size_t, N>, MutViewStorage<T>>;

// // Static shape tensors (compile-time shape)
// template <typename T, std::size_t... Dims>
// using TensorStaticStack = Tensor<T, StaticShape<std::size_t, Dims...>, StackStorage<T, (Dims * ... * 1)>>;

// template <typename T, std::size_t... Dims>
// using TensorStaticHeap = Tensor<T, StaticShape<std::size_t, Dims...>, HeapStorage<T>>;

// template <typename T, std::size_t... Dims>
// using TensorStaticView = Tensor<T, StaticShape<std::size_t, Dims...>, ViewStorage<T>>;

// template <typename T, std::size_t... Dims>
// using TensorStaticMutView = Tensor<T, StaticShape<std::size_t, Dims...>, MutViewStorage<T>>;

// // Common aliases
// template <typename T>
// using Vector = TensorFixedHeap<T, 1>;

// template <typename T>
// using Matrix = TensorFixedHeap<T, 2>;

// template <typename T>
// using Tensor3D = TensorFixedHeap<T, 3>;

// template <typename T>
// using Tensor4D = TensorFixedHeap<T, 4>;

}  // namespace tensor
