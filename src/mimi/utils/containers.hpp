#pragma once

#include <algorithm>
#include <memory>
#include <type_traits>

#include "mimi/utils/print.hpp"

namespace mimi::utils {

/// Adapted from a post from Casey
/// http://stackoverflow.com/a/21028912/273767
/// mentioned in `Note` at
/// http://en.cppreference.com/w/cpp/container/vector/resize
///
/// comments, also from the post:
/// Allocator adaptor that interposes construct() calls to
/// convert value initialization into default initialization.
template<typename Type, typename BaseAllocator = std::allocator<Type>>
class DefaultInitializationAllocator : public BaseAllocator {
  using AllocatorTraits_ = std::allocator_traits<BaseAllocator>;

public:
  template<typename U>
  /// @brief Rebind
  struct rebind {
    using other = DefaultInitializationAllocator<
        U,
        typename AllocatorTraits_::template rebind_alloc<U>>;
  };

  using BaseAllocator::BaseAllocator;

  /// @brief Construct
  /// @tparam U
  /// @param ptr
  template<typename U>
  void
  construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void*>(ptr)) U;
  }
  /// @brief Construct
  /// @tparam U
  /// @tparam ...Args
  /// @param ptr
  /// @param ...args
  template<typename U, typename... Args>
  void construct(U* ptr, Args&&... args) {
    AllocatorTraits_::construct(static_cast<BaseAllocator&>(*this),
                                ptr,
                                std::forward<Args>(args)...);
  }
};

/// @brief short-cut to vector that default initializes
/// @tparam Type
template<typename Type>
using Vector = std::vector<Type, DefaultInitializationAllocator<Type>>;

/// @brief array memory. intended for fast (temporary) use with
/// contiguous memory layout.
/// for 2d-like access, set `stride_` and `is_2d` true
///
/// for non-owning data, use default init and set data and size yourself.
/// size is only used to check bounds in debug mode.
///
/// @tparam Type
template<typename Type, bool is_2d = false, bool is_3d = false>
struct Data {
  Type* data_{nullptr};
  int size_{0};
  int stride0_{1}; // set directly. in case of 3d this should be dim0 * dim1
  int stride1_{1}; // set directly.

  /// @brief can't set size twice
  /// @param n
  void SetSize(int const& n) {
    if (data_) {
      mimi::utils::PrintAndThrowError(
          "data is not empty and can't assign new size.");
    }
    data_ = new Type[n];
    size_ = n;
  }

  Data() = default;
  Data(const int n) { SetSize(n); }

  /// n is size. n * stride is size * stride, don't be confused.
  Data(int n, const int stride) : Data(n), stride0_(stride) {
    assert(n % stride == 0);
  }
  ~Data() { delete[] data_; }

  constexpr Type* data() {
    assert(data_);

    return data_;
  }
  constexpr const Type* data() const {
    assert(data_);

    return data_;
  }
  constexpr int size() { return size_; }

  constexpr Type* begin() {
    assert(data_);

    return data_;
  }

  constexpr Type* end() {
    assert(data_);

    return data_ + size_;
  }

  constexpr void fill(Type const& value) {
    MIMI_FUNC()

    std::fill(data_, data_ + size_, value);
  }

  template<typename IndexType>
  constexpr Type& operator[](const IndexType& i) {
    assert(i < size_);
    assert(data_);

    return data_[i];
  }
  template<typename IndexType>
  constexpr const Type& operator[](const IndexType& i) const {
    assert(i < size_);
    assert(data_);

    return data_[i];
  }
  template<typename IndexType>
  constexpr Type& operator()(const IndexType& i, const IndexType& j) {
    static_assert(is_2d);
    assert(i * stride0_ + j < size_);
    assert(data_);

    return data_[i * stride0_ + j];
  }
  template<typename IndexType>
  constexpr const Type& operator()(const IndexType& i,
                                   const IndexType& j) const {
    static_assert(is_2d);
    assert(i * stride0_ + j < size_);
    assert(data_);

    return data_[i * stride0_ + j];
  }
  template<typename IndexType>
  constexpr Type&
  operator()(const IndexType& i, const IndexType& j, const IndexType& k) {
    static_assert(is_3d);
    assert(i * stride0_ + j * stride1_ + k < size_);
    assert(data_);

    return data_[i * stride0_ + j * stride1_ + k];
  }
  template<typename IndexType>
  constexpr const Type&
  operator()(const IndexType& i, const IndexType& j, const IndexType& k) const {
    static_assert(is_3d);
    assert(i * stride0_ + j * stride1_ + k < size_);
    assert(data_);

    return data_[i * stride0_ + j * stride1_ + k];
  }

  template<typename IndexType>
  constexpr Type* Pointer(const IndexType& i) {
    assert(i < size_);
    assert(data_);

    return &data_[i];
  }
  template<typename IndexType>
  constexpr const Type* Pointer(const IndexType& i) const {
    assert(i < size_);
    assert(data_);

    return &data_[i];
  }
  template<typename IndexType>
  constexpr Type* Pointer(const IndexType& i, const IndexType& j) {
    static_assert(is_2d);
    assert(i * stride0_ + j < size_);
    assert(data_);

    return &data_[i * stride0_ + j];
  }
  template<typename IndexType>
  constexpr const Type* Pointer(const IndexType& i, const IndexType& j) const {
    static_assert(is_2d);
    assert(i * stride0_ + j < size_);
    assert(data_);

    return &data_[i * stride0_ + j];
  }
  template<typename IndexType>
  constexpr Type*
  Pointer()(const IndexType& i, const IndexType& j, const IndexType& k) const {
    static_assert(is_3d);
    assert(i * stride0_ + j * stride1_ + k < size_);
    assert(data_);

    return &data_[i * stride0_ + j * stride1_ + k];
  }
  template<typename IndexType>
  constexpr Type*
  Pointer()(const IndexType& i, const IndexType& j, const IndexType& k) {
    static_assert(is_3d);
    assert(i * stride0_ + j * stride1_ + k < size_);
    assert(data_);

    return &data_[i * stride0_ + j * stride1_ + k];
  }
};

} // namespace mimi::utils
