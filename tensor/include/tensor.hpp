#pragma once

// Main tensor header - includes all tensor types

#include "tensor_array.hpp"
#include "tensor_array_mut_view.hpp"
#include "tensor_array_view.hpp"
#include "tensor_fixed.hpp"
#include "tensor_fixed_mut_view.hpp"
#include "tensor_fixed_view.hpp"
#include "tensor_static.hpp"
#include "tensor_static_mut_view.hpp"
#include "tensor_static_view.hpp"

namespace infer {

// Type aliases for common tensor types
template <typename T>
using Vector = TensorFixed<T, 1>;

template <typename T>
using Matrix = TensorFixed<T, 2>;

template <typename T>
using Tensor3D = TensorFixed<T, 3>;

template <typename T>
using Tensor4D = TensorFixed<T, 4>;

// View type aliases
template <typename T>
using VectorView = TensorFixedView<T, 1>;

template <typename T>
using MatrixView = TensorFixedView<T, 2>;

template <typename T>
using VectorMutView = TensorFixedMutView<T, 1>;

template <typename T>
using MatrixMutView = TensorFixedMutView<T, 2>;

}  // namespace infer
