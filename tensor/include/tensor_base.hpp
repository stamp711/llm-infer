#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <memory>

#include "tensor_traits.hpp"

namespace infer {

constexpr std::size_t TENSOR_ALIGNMENT = 64;

// Level 1: Common read-only operations for all tensor types
template <typename Derived>
    requires TensorTraits<TensorInnerTypes<Derived>>
class TensorBaseCommon {
   public:
    using inner_types = TensorInnerTypes<Derived>;
    using value_type = typename inner_types::value_type;
    using size_type = typename inner_types::size_type;
    using shape_type = typename inner_types::shape_type;

    [[nodiscard]] const value_type* data() const noexcept { return derived().data_impl(); }
    [[nodiscard]] const shape_type& shape() const noexcept { return derived().shape_impl(); }
    [[nodiscard]] size_type size() const noexcept { return derived().size_impl(); }
    [[nodiscard]] size_type rank() const noexcept { return derived().rank_impl(); }

    [[nodiscard]] const value_type& operator[](size_type idx) const { return data()[idx]; }

   protected:
    TensorBaseCommon() = default;
    ~TensorBaseCommon() = default;
    TensorBaseCommon(const TensorBaseCommon&) = default;
    TensorBaseCommon& operator=(const TensorBaseCommon&) = default;
    TensorBaseCommon(TensorBaseCommon&&) = default;
    TensorBaseCommon& operator=(TensorBaseCommon&&) = default;

   private:
    [[nodiscard]] Derived& derived() noexcept { return static_cast<Derived&>(*this); }
    [[nodiscard]] const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }
};

// Level 2: Adds mutable operations
template <typename Derived>
class MutableTensorBase : public TensorBaseCommon<Derived> {
   public:
    using base_type = TensorBaseCommon<Derived>;
    using typename base_type::size_type;
    using typename base_type::value_type;

    [[nodiscard]] value_type* data() noexcept { return this->derived().data_impl(); }
    using base_type::data;  // Keep const version available

    [[nodiscard]] value_type& operator[](size_type idx) { return data()[idx]; }
    using base_type::operator[];  // Keep const version available

    void fill(value_type value) { std::fill_n(data(), this->size(), value); }
    void zeros() { fill(value_type(0)); }
    void ones() { fill(value_type(1)); }

   protected:
    MutableTensorBase() = default;
    ~MutableTensorBase() = default;
    MutableTensorBase(const MutableTensorBase&) = default;
    MutableTensorBase& operator=(const MutableTensorBase&) = default;
    MutableTensorBase(MutableTensorBase&&) = default;
    MutableTensorBase& operator=(MutableTensorBase&&) = default;

   private:
    [[nodiscard]] Derived& derived() noexcept { return static_cast<Derived&>(*this); }
    [[nodiscard]] const Derived& derived() const noexcept { return static_cast<const Derived&>(*this); }
};

// Level 2b: Alias for read-only tensors
template <typename Derived>
using ReadOnlyTensorBase = TensorBaseCommon<Derived>;

// Level 3a: Base for owning tensors with runtime shape management
template <typename Derived>
class OwningStridedTensorBase : public MutableTensorBase<Derived> {
   public:
    using base_type = MutableTensorBase<Derived>;
    using typename base_type::shape_type;
    using typename base_type::size_type;
    using typename base_type::value_type;

   protected:
    // Custom deleter for aligned memory
    struct AlignedDeleter {
        void operator()(value_type* ptr) const { std::free(ptr); }  // NOLINT
    };

    std::unique_ptr<value_type[], AlignedDeleter> data_;  // NOLINT - owned data storage
    shape_type shape_;                                    // Runtime shape
    size_type size_ = 0;                                  // Total number of elements

    void allocate() {
        if (size_ == 0) return;
        void* ptr = std::aligned_alloc(TENSOR_ALIGNMENT, size_ * sizeof(value_type));  // NOLINT
        if (ptr == nullptr) throw std::bad_alloc();
        data_.reset(static_cast<value_type*>(ptr));
    }

    void compute_size() {
        size_ = 1;
        for (auto dim : shape_) size_ *= dim;
    }

    OwningStridedTensorBase() = default;
    ~OwningStridedTensorBase() = default;
    OwningStridedTensorBase(const OwningStridedTensorBase&) = default;
    OwningStridedTensorBase& operator=(const OwningStridedTensorBase&) = default;
    OwningStridedTensorBase(OwningStridedTensorBase&&) = default;
    OwningStridedTensorBase& operator=(OwningStridedTensorBase&&) = default;

   public:
    // CRTP implementations for owning tensors
    [[nodiscard]] value_type* data_impl() noexcept { return data_.get(); }
    [[nodiscard]] const value_type* data_impl() const noexcept { return data_.get(); }
    [[nodiscard]] const shape_type& shape_impl() const noexcept { return shape_; }
    [[nodiscard]] size_type size_impl() const noexcept { return size_; }
};

// Level 3b: Base for non-owning mutable views with runtime shape
template <typename Derived>
class MutableViewStridedTensorBase : public MutableTensorBase<Derived> {
   public:
    using base_type = MutableTensorBase<Derived>;
    using typename base_type::shape_type;
    using typename base_type::size_type;
    using typename base_type::value_type;

   protected:
    value_type* data_ptr_ = nullptr;  // Non-owning pointer to mutable data
    shape_type shape_;                // Runtime shape
    size_type size_ = 0;              // Total number of elements

    void compute_size() {
        size_ = 1;
        for (auto dim : shape_) size_ *= dim;
    }

    MutableViewStridedTensorBase() = default;
    ~MutableViewStridedTensorBase() = default;
    MutableViewStridedTensorBase(const MutableViewStridedTensorBase&) = default;
    MutableViewStridedTensorBase& operator=(const MutableViewStridedTensorBase&) = default;
    MutableViewStridedTensorBase(MutableViewStridedTensorBase&&) = default;
    MutableViewStridedTensorBase& operator=(MutableViewStridedTensorBase&&) = default;

   public:
    // CRTP implementations for mutable views
    [[nodiscard]] value_type* data_impl() noexcept { return data_ptr_; }
    [[nodiscard]] const value_type* data_impl() const noexcept { return data_ptr_; }
    [[nodiscard]] const shape_type& shape_impl() const noexcept { return shape_; }
    [[nodiscard]] size_type size_impl() const noexcept { return size_; }
};

// Level 3c: Base for non-owning read-only views with runtime shape
template <typename Derived>
class ReadOnlyViewStridedTensorBase : public ReadOnlyTensorBase<Derived> {
   public:
    using base_type = ReadOnlyTensorBase<Derived>;
    using typename base_type::shape_type;
    using typename base_type::size_type;
    using typename base_type::value_type;

   protected:
    const value_type* data_ptr_ = nullptr;  // Non-owning pointer to const data
    shape_type shape_;                      // Runtime shape
    size_type size_ = 0;                    // Total number of elements

    void compute_size() {
        size_ = 1;
        for (auto dim : shape_) size_ *= dim;
    }

    ReadOnlyViewStridedTensorBase() = default;
    ~ReadOnlyViewStridedTensorBase() = default;
    ReadOnlyViewStridedTensorBase(const ReadOnlyViewStridedTensorBase&) = default;
    ReadOnlyViewStridedTensorBase& operator=(const ReadOnlyViewStridedTensorBase&) = default;
    ReadOnlyViewStridedTensorBase(ReadOnlyViewStridedTensorBase&&) = default;
    ReadOnlyViewStridedTensorBase& operator=(ReadOnlyViewStridedTensorBase&&) = default;

   public:
    // CRTP implementations for read-only views
    [[nodiscard]] const value_type* data_impl() const noexcept { return data_ptr_; }
    [[nodiscard]] const shape_type& shape_impl() const noexcept { return shape_; }
    [[nodiscard]] size_type size_impl() const noexcept { return size_; }
};

}  // namespace infer
