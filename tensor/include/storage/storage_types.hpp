#pragma once

#include <array>
#include <cstddef>
#include <vector>

#include "storage_concepts.hpp"

namespace tensor {

// Stack-based storage using std::array
template <typename T, std::size_t N>
class StackStorage {
   public:
    using value_type = T;
    using index_type = std::size_t;

   private:
    alignas(64) std::array<T, N> data_;

   public:
    StackStorage() = default;

    explicit StackStorage(const T& fill_value) { data_.fill(fill_value); }

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }
    [[nodiscard]] constexpr index_type size() const noexcept { return N; }
    [[nodiscard]] constexpr bool empty() const noexcept { return N == 0; }
    [[nodiscard]] constexpr index_type capacity() const noexcept { return N; }

    // Array-like interface
    [[nodiscard]] T& operator[](index_type idx) noexcept { return data_[idx]; }
    [[nodiscard]] const T& operator[](index_type idx) const noexcept { return data_[idx]; }
};
static_assert(TensorStorage<StackStorage<float, 100>>);
static_assert(MutableTensorStorage<StackStorage<float, 100>>);

// Heap-based storage using std::vector
template <typename T, typename IndexT = std::size_t>
class HeapStorage {
   public:
    using value_type = T;
    using index_type = IndexT;

   private:
    std::vector<T> data_;

   public:
    HeapStorage() = default;

    explicit HeapStorage(index_type size) : data_(size) {}

    HeapStorage(index_type size, const T& fill_value) : data_(size, fill_value) {}

    [[nodiscard]] T* data() noexcept { return data_.data(); }
    [[nodiscard]] const T* data() const noexcept { return data_.data(); }
    [[nodiscard]] index_type size() const noexcept { return data_.size(); }
    [[nodiscard]] bool empty() const noexcept { return data_.empty(); }
    [[nodiscard]] index_type capacity() const noexcept { return data_.capacity(); }

    void resize(index_type size) { data_.resize(size); }
    void resize(index_type size, const T& value) { data_.resize(size, value); }
    void reserve(index_type capacity) { data_.reserve(capacity); }

    // Array-like interface
    [[nodiscard]] T& operator[](index_type idx) noexcept { return data_[idx]; }
    [[nodiscard]] const T& operator[](index_type idx) const noexcept { return data_[idx]; }
};
static_assert(TensorStorage<HeapStorage<float>>);
static_assert(MutableTensorStorage<HeapStorage<float>>);
static_assert(ResizableTensorStorage<HeapStorage<float>>);

// Non-owning view storage for const data
template <typename ValueT, typename IndexT = std::size_t>
class ViewStorage {
   public:
    using value_type = ValueT;
    using index_type = IndexT;

   private:
    const ValueT* data_ptr_ = nullptr;
    index_type size_ = 0;

   public:
    ViewStorage() = default;

    ViewStorage(const ValueT* ptr, index_type size) : data_ptr_(ptr), size_(size) {}

    [[nodiscard]] const ValueT* data() const noexcept { return data_ptr_; }
    [[nodiscard]] index_type size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    // Array-like interface
    [[nodiscard]] const ValueT& operator[](index_type idx) const noexcept { return data_ptr_[idx]; }
};
static_assert(TensorStorage<ViewStorage<float>>);

// Non-owning view storage for mutable data
template <typename ValueT, typename IndexT = std::size_t>
class MutViewStorage {
   public:
    using value_type = ValueT;
    using index_type = IndexT;

   private:
    ValueT* data_ptr_ = nullptr;
    index_type size_ = 0;

   public:
    MutViewStorage() = default;

    MutViewStorage(ValueT* ptr, index_type size) : data_ptr_(ptr), size_(size) {}

    [[nodiscard]] ValueT* data() noexcept { return data_ptr_; }
    [[nodiscard]] const ValueT* data() const noexcept { return data_ptr_; }
    [[nodiscard]] index_type size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }

    // Array-like interface
    [[nodiscard]] ValueT& operator[](index_type idx) noexcept { return data_ptr_[idx]; }
    [[nodiscard]] const ValueT& operator[](index_type idx) const noexcept { return data_ptr_[idx]; }
};
static_assert(MutableTensorStorage<MutViewStorage<float>>);

}  // namespace tensor
