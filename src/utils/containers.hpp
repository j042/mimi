#pragma once

#include <memory>
#include <type_traits>

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
/// for 2d-like access, set `stride_`
/// @tparam Type
template<typename Type>
struct Data {
  Type* data_;
  const int size_{0};
  int stride_{1};

  Data(const int n) : data_(new Type[n]), size_(n) {}
  Data(int n, const int stride) : Data(n), stride_(stride) {}
  ~Data() { delete[] data_; }

  template<typename IndexType>
  constexpr Type& operator[](const IndexType& i) {
    return data_[i];
  }
  template<typename IndexType>
  constexpr const Type& operator[](const IndexType& i) const {
    return data_[i];
  }
  template<typename IndexType>
  constexpr Type& operator[](const IndexType& i, const IndexType& j) {
    return data_[i * stride_ + j];
  }
  template<typename IndexType>
  constexpr const T& operator[](const IndexType& i, const IndexType& j) const {
    return data_[i * stride_ + j];
  }

  template<typename IndexType>
  constexpr Type* Pointer(const IndexType& i) {
    return &data_[i];
  }
  template<typename IndexType>
  constexpr const Type* Pointer(const IndexType& i) const {
    return &data_[i];
  }
  template<typename IndexType>
  constexpr Type* Pointer(const IndexType& i, const IndexType& j) {
    return &data_[i * stride_ + j];
  }
  template<typename IndexType>
  constexpr const Type* Pointer(const IndexType& i, const IndexType& j) const {
    return &data_[i * stride_ + j];
  }
};

} // namespace mimi::utils
