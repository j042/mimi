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

template<typename MatrixType, typename VectorType, typename ScalarType>
struct DataSeries {
  Vector<MatrixType> matrices;
  Vector<VectorType> vectors;
  Vector<ScalarType> scalars;
};

/// @brief Fully dynamic array that can view another data. Equipped with basic
/// math operations.
/// @tparam DataType
/// @tparam dim
template<typename DataType, int dim = 1, typename IndexType = int>
class Data {
  static_assert(dim > 0, "dim needs to be positive value bigger than zero.");
  static_assert(std::is_integral_v<IndexType>,
                "IndexType should be an integral type");

public:
  using ShapeType_ = std::array<IndexType, dim>;
  using StridesType_ = std::array<IndexType, dim - 1>;
  static constexpr const IndexType kDim = static_cast<IndexType>(dim);

  using DataType_ = DataType;
  using IndexType_ = IndexType;

  // std container like types
  using value_type = DataType;
  using size_type = IndexType;

protected:
  bool own_data_{false};
  /// @brief beginning of the array pointer that this object manages
  DataType* data_{nullptr};

  /// @brief size of this Array
  IndexType size_{};

  /// @brief strides in case this is a higher dim. last entry should be the same
  /// as size_
  StridesType_ strides_;

  /// @brief shape of the array
  ShapeType_ shape_;

  /// @brief helper to find id offset
  /// @tparam ...Ts
  /// @param index
  /// @param counter
  /// @param ...id
  template<typename... Ts>
  constexpr void ComputeRemainingOffset(IndexType& index,
                                        IndexType& counter,
                                        const IndexType& id0,
                                        const Ts&... id) const {
    // last element should be just added
    if constexpr (sizeof...(Ts) == 0) {
      index += id0;
      // intentionally skipping
      // ++counter;
      return;
    }

    // add strides * id
    index += strides_[counter++] * id0;

    // do the rest
    if constexpr (sizeof...(Ts) > 0) {
      ComputeRemainingOffset(index, counter, id...);
    }
  }

public:
  constexpr DataType* data() {
    assert(data_);
    return data_;
  }

  constexpr const DataType* data() const {
    assert(data_);
    return data_;
  }

  constexpr DataType* begin() {
    assert(data_);
    return data_;
  }

  constexpr const DataType* begin() const {
    assert(data_);
    return data_;
  }

  constexpr DataType* end() {
    assert(data_);
    assert(size_ > 0);
    return data_ + size_;
  }

  constexpr const DataType* end() const {
    assert(data_);
    assert(size_ > 0);
    return data_ + size_;
  }

  constexpr IndexType size() const { return size_; }
  constexpr IndexType Size() const { return size_; }

  constexpr void DestroyData() {
    if (own_data_ && data_) {
      delete[] data_;
    }

    data_ = nullptr;
    own_data_ = false;
  }

  constexpr void SetData(DataType* data_pointer) {
    DestroyData();
    data_ = data_pointer;
  }

  constexpr DataType* GetData() {
    assert(data_);
    return data_;
  }

  constexpr const DataType* GetData() const {
    assert(data_);
    return data_;
  }

  /// @brief reallocates space. After this call, own_data_ should be true
  /// @param size
  constexpr void Reallocate(const IndexType& size) {
    // destroy and reallocate space
    DestroyData();
    data_ = new DataType[size];
    own_data_ = true;

    // set size - don't forget to set shape in case this is multi-dim array
    size_ = size;

    // set shape if this is a 1d array
    if constexpr (dim == 1) {
      shape_[0] = size;
    }
  }

  template<typename... Ts>
  constexpr void SetShape(const IndexType& shape0, const Ts&... shape) {
    static_assert(sizeof...(Ts) == static_cast<std::size_t>(dim - 1),
                  "shape should match the dim");

    // early exit in case of dim 1
    if constexpr (sizeof...(Ts) == 0) {
      size_ = shape0;
      shape_[0] = shape0;
      return;
    }

    // (re)initialize shape_
    shape_ = {shape0, shape...};

    // (re)initialize  strides_
    strides_ = {shape...};

    // reverse iter and accumulate
    IndexType sub_total{1};
    typename StridesType_::reverse_iterator r_stride_iter = strides_.rbegin();
    for (; r_stride_iter != strides_.rend(); ++r_stride_iter) {
      // accumulate
      sub_total *= *r_stride_iter;
      // assign
      *r_stride_iter = sub_total;
    }

    // at this point, sub total has accumulated shape values except shape0
    // set size
    size_ = sub_total * shape0;
  }

  /// @brief default. use SetData() and SetShape<false>
  constexpr Data() = default;

  /// @brief basic array ctor
  /// @tparam ...Ts
  /// @param ...shape
  template<typename... Ts>
  constexpr Data(const Ts&... shape) {
    SetShape(shape...);
    Reallocate(size_);
  }

  /// @brief data wrapping array
  /// @tparam ...Ts
  /// @param data_pointer
  /// @param ...shape
  template<typename... Ts>
  constexpr Data(DataType* data_pointer, const Ts&... shape) {
    SetData(data_pointer);
    SetShape(shape...);
  }

  /// @brief copy ctor
  /// @param other
  constexpr Data(const Data& other) {
    // memory alloc
    Reallocate(other.size_);
    // copy data
    std::copy_n(other.data(), other.size_, data_);
    // copy shape
    shape_ = other.shape_;
    // copy strides
    strides_ = other.strides_;
  }

  /// @brief move ctor
  /// @param other
  constexpr Data(Data&& other) {
    own_data_ = other.own_data_;
    data_ = std::move(other.data_);
    size_ = std::move(other.size_);
    strides_ = std::move(other.strides_);
    shape_ = std::move(other.shape_);
    other.own_data_ = false;
  }

  /// @brief copy assignment - need to make sure that you have enough space
  /// @param rhs
  /// @return
  constexpr Data& operator=(const Data& rhs) {
    // size check is crucial, so runtime check
    if (size_ != rhs.size_) {
      throw std::runtime_error(
          "Data::operator=(const Data&) - size mismatch between rhs");
    }
    std::copy_n(rhs.data(), rhs.size(), data_);

    // copy shape and stride info
    shape_ = rhs.Shape();
    strides_ = rhs.Strides();

    return *this;
  }

  /// @brief move assignment. currently same as ctor.
  /// @param rhs
  /// @return
  constexpr Data& operator=(Data&& rhs) {
    DestroyData();

    own_data_ = rhs.own_data_;
    data_ = std::move(rhs.data_);
    size_ = std::move(rhs.size_);
    strides_ = std::move(rhs.strides_);
    shape_ = std::move(rhs.shape_);
    rhs.own_data_ = false;

    return *this;
  }

  /// @brief
  ~Data() { DestroyData(); }

  /// @brief Returns const shape object
  /// @return
  constexpr const ShapeType_& Shape() const { return shape_; }

  /// @brief Returns const strides object
  /// @return
  constexpr const StridesType_& Strides() const { return strides_; }

  /// @brief flat indexed array access
  /// @param index
  /// @return
  constexpr DataType& operator[](const IndexType& index) {
    assert(index > -1 && index < size_);

    return data_[index];
  }

  /// @brief flat indexed array access
  /// @param index
  /// @return
  constexpr const DataType& operator[](const IndexType& index) const {
    assert(index > -1 && index < size_);

    return data_[index];
  }

  /// @brief multi-indexed array access
  /// @tparam ...Ts
  /// @param id0
  /// @param ...id
  /// @return
  template<typename... Ts>
  constexpr DataType& operator()(const IndexType& id0, const Ts&... id) {
    static_assert(sizeof...(Ts) == static_cast<std::size_t>(dim - 1),
                  "number of index should match dim");

    if constexpr (dim == 1) {
      return operator[](id0);
    }

    IndexType final_id{id0 * strides_[0]}, i{1};
    if constexpr (sizeof...(Ts) > 0) {
      ComputeRemainingOffset(final_id, i, id...);
    }

    assert(final_id > -1 && final_id < size_);

    return data_[final_id];
  }

  /// @brief multi-indexed array access
  /// @tparam ...Ts
  /// @param id0
  /// @param ...id
  /// @return
  template<typename... Ts>
  constexpr const DataType& operator()(const IndexType& id0,
                                       const Ts&... id) const {
    static_assert(sizeof...(Ts) == static_cast<std::size_t>(dim - 1),
                  "number of index should match dim");

    if constexpr (dim == 1) {
      return operator[](id0);
    }

    IndexType final_id{id0 * strides_[0]}, i{1};
    if constexpr (sizeof...(Ts) > 0) {

      ComputeRemainingOffset(final_id, i, id...);
    }

    assert(final_id > -1 && final_id < size_);

    return data_[final_id];
  }

  /// @brief this[:] = v
  /// @param v
  constexpr Data& Fill(const DataType& v) {
    assert(data_);

    std::fill_n(data_, size_, v);

    return *this;
  }

  /// @brief this[i] = a[i]
  /// @param a
  /// @return
  constexpr Data& Copy(const DataType* a) {
    assert(data_);

    std::copy_n(a, size_, data_);

    return *this;
  }

  /// @brief copies from a and this will own the data
  /// @param a
  /// @return
  constexpr Data& OwnCopy(const Data& a) {
    // this one doesn't need to have an "active" data_

    Reallocate(a.size());

    // then copy assign
    operator=(a);

    return *this;
  }

  /// @brief this[i] += a[i]
  /// @tparam Iterable
  /// @param a
  template<typename Iterable>
  constexpr Data& Add(const Iterable& a) {
    assert(a.size() == size_);
    assert(data_);

    for (IndexType i{}; i < size_; ++i) {
      data_[i] += a[i];
    }
    return *this;
  }

  /// @brief this[i] += a * b[i]
  /// @param a
  /// @param b
  /// @return
  constexpr Data& Add(const DataType_& a, const DataType* b) {
    assert(data_);

    for (IndexType i{}; i < size_; ++i) {
      data_[i] += a * b[i];
    }
    return *this;
  }

  /// @brief this[i] += abs(a * b[i])
  /// @param a
  /// @param b
  /// @return
  constexpr Data& AddAbs(const DataType_& a, const DataType* b) {
    assert(data_);

    for (IndexType i{}; i < size_; ++i) {
      data_[i] += std::abs(a * b[i]);
    }
    return *this;
  }

  /// @brief this[i] *= a * this[i]
  /// @param a
  constexpr Data& Multiply(const DataType& a) {
    assert(data_);

    for (DataType* d = begin(); d != end();) {
      *d++ *= a;
    }

    return *this;
  }

  /// @brief this[i] = a * b[i]. Must set size and data beforehand.
  /// @param a
  /// @param v
  constexpr Data& MultiplyAssign(const DataType& a, const DataType* b) {
    assert(data_);

    for (IndexType i{}; i < size_; ++i) {
      data_[i] = a * b[i];
    }

    return *this;
  }

  /// @brief this[i] -= a[i]
  /// @tparam Iterable
  /// @param a
  template<typename Iterable>
  constexpr Data& Subtract(const Iterable& a) {
    assert(a.size() == size_);
    assert(data_);

    for (IndexType i{}; i < size_; ++i) {
      data_[i] -= a[i];
    }

    return *this;
  }

  /// @brief this[i] -= a[i]
  /// @tparam Iterable
  /// @param a
  constexpr Data& Subtract(const DataType* a) {
    assert(data_);

    for (IndexType i{}; i < size_; ++i) {
      data_[i] -= a[i];
    }

    return *this;
  }

  /// @brief c = (this) dot (a).
  /// @tparam Iterable
  /// @param a
  /// @return
  template<typename Iterable>
  constexpr DataType InnerProduct(const Iterable& a) const {
    static_assert(dim == 1, "inner product is only applicable for dim=1 Data.");
    assert(a.size() == size_);
    assert(data_);

    DataType dot{};
    for (IndexType i{}; i < size_; ++i) {
      dot += a[i] * data_[i];
    }
    return dot;
  }

  /// @brief L2 norm
  /// @return
  constexpr DataType NormL2() {
    std::remove_const_t<DataType> norm{};
    for (IndexType i{}; i < size_; ++i) {
      const DataType& data_i = data_[i];
      norm += data_i * data_i;
    }
    return std::sqrt(norm);
  }
};

template<typename T, typename IndexType>
void AddDiagonal(T* data, const T fac, const IndexType dim) {
  MIMI_FUNC()

  for (IndexType i{}; i < dim; ++i) {
    data[(dim + 1) * i] += fac;
  }
}

template<typename T>
Vector<T> Arange(const T from, const T to) {
  MIMI_FUNC()
  static_assert(std::is_integral_v<T>, "T should be an integral type.");
  assert(to > from);
  Vector<T> arange(to - from);
  // same as std::iota
  for (T i{from}, j{}; i < to; ++i, ++j) {
    arange[j] = i;
  }
  return arange;
}

} // namespace mimi::utils
