/*
MIT License

Copyright (c) 2022 zwar@ilsb.tuwien.ac.at

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#pragma once

// c++ library
#include <array>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <mfem.hpp>

// custom container
#include "mimi/utils/containers.hpp"

namespace mimi::utils {

/*!
 * @class ADScalar
 *
 * This class implements a simple data type for automatic differentiation. The
 * basic mathematical operations have been overloaded using appropriate rules of
 * differentiation in order to determine the value of a computation as well as
 * the derivative with respect to the input at the same time.
 *
 * This implementation is based on and largely inspired by @danielwolff1's
 * implementation in campiga, modified to be used at runtime. This slows down
 * the code, but provides more flexibility.
 *
 * Supported operations:
 *  - Addition
 *  - Subtraction
 *  - Multiplication
 *  - Division
 *  - log, log10 and exp
 *  - sqrt, power
 *  - abs
 *  - Boolean operations
 *  - sin, cos, tan, asin, acos, atan
 *
 * @tparam Scalar      The intrinsic data type for storing the value and the
 *                     derivative components
 * @tparam number_of_derivatives
 */
template<typename Scalar, int number_of_derivatives = 0>
class ADScalar {
public:
  /// Alias for the intrinsic scalar data type
  using Scalar_ = Scalar;

  /// Alias for the indexing
  using IndexingType_ = int;

  /// Alias for the data type of this class
  using ADT_ = ADScalar<Scalar, number_of_derivatives>;

  /// Alias for the data type storing the gradient
  using DerivType_ =
      std::conditional_t<(number_of_derivatives < 0),
                         mimi::utils::Data<Scalar_>,
                         std::array<Scalar_, number_of_derivatives>>;

private:
  /*!
   * Constructor which initializes all member variables directly.
   * This is only meant to be used by this class internally.
   *
   * @param v    The value
   * @param d    The gradient
   */
  ADScalar(const Scalar_& v, const DerivType_& d) : v_(v), d_(d) {}

  /*!
   * Stores the result of the computation carried out with the
   * i-th component of vector \f$ x \f$
   */
  Scalar v_;

  /*!
   * Stores the gradient \f$ \nabla_{x_i} f(x) \f$ of the computation carried
   * out with respect to component \f$ x_i \f$
   */
  DerivType_ d_;

public:
  constexpr Scalar& ValueReadWrite() { return v_; }
  constexpr DerivType_& DerivativeReadWrite() { return d_; }

  /// Default Copy Constructor
  constexpr ADScalar(const ADT_& rhs) = default;

  /// Default Move Constructor
  constexpr ADScalar(ADT_&& rhs) = default;

  /// Empty constructor which initializes all member variables to zero
  ADScalar() : v_{}, d_{} {}

  /// Scalar Constructor without Derivative
  template<int dummy = number_of_derivatives,
           typename std::enable_if_t<dummy == 0>* = nullptr>
  ADScalar(const Scalar_& value, const IndexingType_& n_derivatives)
      : v_{value} {
    d_.Reallocate(n_derivatives);
    d_.Fill(Scalar_{});
  } // d_{size} initializes to Scalar{}

  /// Scalar Constructor without Derivative
  template<int dummy = number_of_derivatives,
           typename std::enable_if_t<dummy != 0>* = nullptr>
  ADScalar(const Scalar_& value)
      : v_{value},
        d_{} {} // d_{size} initializes to Scalar{}

  /// Define a destructor
  ~ADScalar() = default;

  /*!
   * Constructor for user initialization of an instance. It initializes an
   * instance with the value \f$ x_i \f$ for which the computation is supposed
   * to be carried out and sets its derivative to the canonical basis vector \f$
   * e_i \f$, which corresponds to the component with respect to which the
   * derivative is supposed to be computed.
   */
  template<int dummy = number_of_derivatives,
           typename std::enable_if_t<dummy == 0>* = nullptr>
  ADScalar(const Scalar& value,
           const IndexingType_& n_derivatives,
           const IndexingType_ active_component)
      : v_{value} {
    d_.Reallocate(n_derivatives);
    d_.Fill(Scalar_{})
        // Check if requested derivative is in range
        assert(active_component < d_.size());
    SetActiveComponent(active_component);
  }

  /*!
   * Constructor for user initialization of an instance. It initializes an
   * instance with the value \f$ x_i \f$ for which the computation is supposed
   * to be carried out and sets its derivative to the canonical basis vector \f$
   * e_i \f$, which corresponds to the component with respect to which the
   * derivative is supposed to be computed.
   */
  template<int dummy = number_of_derivatives,
           typename std::enable_if_t<dummy != 0>* = nullptr>
  ADScalar(const Scalar& value, const IndexingType_ active_component)
      : v_{value},
        d_{} {
    assert(active_component < d_.size());
    SetActiveComponent(active_component);
  }

  /*!
   * Overload of assign operator
   *
   * If second derivatives are computed (through nesting ADScalars), this
   * overload is necessary to assign ADTs by passing scalars.
   */
  template<typename BaseType>
  ADScalar& operator=(const BaseType& base) {
    *this = ADScalar(base);
    return *this;
  }

  ADScalar& operator=(const ADScalar& t) = default;

  /** @defgroup GetterSetters Getter and Setter Methods
   * Setter methods for values and derivatives
   * @{
   */

  /*!
   * Marks the current variable as the component-th active variable of a vector
   * by first filling the derivative vector with zeros and then setting the
   * provided component to one.
   * @param component    The component in range {0,...,n_derivs-1} which is
   * represented by this variable
   * @note This method is meant for initialization and should not be used within
   *       mathematical computations
   */
  void SetActiveComponent(const int component) {
    std::fill(d_.begin(), d_.end(), Scalar_{});
    d_[component] = 1.0;
  }

  /// Getter Function
  constexpr Scalar GetValue() const { return v_; }

  /// Getter derivative
  constexpr DerivType_ GetDerivatives() const { return d_; }

  constexpr Scalar GetDerivatives(const int i) const {
    MIMI_FUNC()
    return d_[i];
  }

  /*!
   * Returns the spatial dimensionality of vector \f$ x \f$
   * @return     The value of the template parameter n_derivs
   */
  constexpr IndexingType_ GetNumberOfDerivatives() const { return d_.size(); }

  /** @} */ // End of Getters and Setters

  /** @defgroup Operators Overloaded Basic Operators
   * All basic operations that change the value or the derivative of the object
   * and that can be overloaded as class members
   * @{
   */

  /// Addition with ADScalar
  constexpr ADT_ operator+(const ADT_& b) const;

  /// Addition and self assignment with ADScalar
  constexpr ADT_& operator+=(const ADT_& b);

  /// Addition with Scalar
  constexpr ADT_ operator+(const Scalar& b) const;

  /// Addition and self assignment with Scalar
  constexpr ADT_& operator+=(const Scalar& b);

  /// Negate the value of ADScalar
  constexpr ADT_ operator-() const;

  /// Subtraction with ADScalar
  constexpr ADT_ operator-(const ADT_& b) const;

  /// Subtraction and self assignment with ADScalar
  constexpr ADT_& operator-=(const ADT_& b);

  /// Subtraction with Scalar
  constexpr ADT_ operator-(const Scalar& b) const;

  /// Subtraction and self assignment with Scalar
  constexpr ADT_& operator-=(const Scalar& b);

  /// Multiplication with ADScalar
  constexpr ADT_ operator*(const ADT_& b) const;

  /// Multiplication and self assignment with ADScalar
  constexpr ADT_& operator*=(const ADT_& b);

  /// Multiplication with Scalar
  constexpr ADT_ operator*(const Scalar& b) const;

  /// Multiplication and self assignment with Scalar
  constexpr ADT_& operator*=(const Scalar& b);

  /// Division by ADScalar
  constexpr ADT_ operator/(const ADT_& b) const;

  /// Division and self assignment by ADScalar
  constexpr ADT_& operator/=(const ADT_& b);

  /// Division by Scalar
  constexpr ADT_ operator/(const Scalar& b) const;

  /// Division and self assignment by Scalar
  constexpr ADT_& operator/=(const Scalar& b);

  /** @} */ // End of Basic operations

  /** @defgroup BoolOperators Boolean Operators
   * All boolean operations that change that can be overloaded as class members
   * @{
   */

  /// Greater operator
  constexpr bool operator>(const Scalar& b) const { return v_ > b; }

  /// Greater operator
  constexpr bool operator>(const ADT_& b) const { return v_ > b.v_; }

  /// Greater equal operator
  constexpr bool operator>=(const Scalar& b) const { return v_ >= b; }

  /// Greater equal operator
  constexpr bool operator>=(const ADT_& b) const { return v_ >= b.v_; }

  /// Smaller operator
  constexpr bool operator<(const Scalar& b) const { return v_ < b; }

  /// Smaller operator, delegate to operator>=(const ADT_ &adt)
  constexpr bool operator<(const ADT_& b) const { return v_ < b; }

  /// Smaller equal operator
  constexpr bool operator<=(const Scalar& b) const { return v_ <= b; }

  /// Smaller equal operator
  constexpr bool operator<=(const ADT_& b) const { return v_ <= b; }

  /// Equal operator
  constexpr bool operator==(const Scalar& b) const { return v_ == b; }

  /// Equal operator for ADScalar (assuming only value considered)
  constexpr bool operator==(const ADT_& b) const { return v_ == b.v_; }

  /// Inequality operator
  constexpr bool operator!=(const Scalar& b) const { return v_ != b; }

  /// Inequality operator for ADScalar (assuming only value considered)
  constexpr bool operator!=(const ADT_& b) const { return v_ != b; }

  /** @} */ // End of Boolean Operators

  /** @defgroup friendInjections Friend Injections
   * All operations were the order prohibits the use of class member functions
   * @{
   */

  /// Simple Default Output stream overload
  template<typename ScalarF, int number_of_derivativesF>
  friend std::ostream&
  operator<<(std::ostream& os,
             const ADScalar<ScalarF, number_of_derivativesF>& a);

  /// Addition
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  operator+(const ScalarF& a,
            const ADScalar<ScalarF, number_of_derivativesF>& b);

  /// Subtraction
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  operator-(const ScalarF& a,
            const ADScalar<ScalarF, number_of_derivativesF>& b);

  /// Multiplication
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  operator*(const ScalarF& a,
            const ADScalar<ScalarF, number_of_derivativesF>& b);

  /// Division
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  operator/(const ScalarF& a,
            const ADScalar<ScalarF, number_of_derivativesF>& b);

  /// Natural exponent of ADScalar (e.g. \f$ \exp{x_i} \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  exp(const ADScalar<ScalarF, number_of_derivativesF>& exponent);

  /// Absolute Value
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  abs(const ADScalar<ScalarF, number_of_derivativesF>& base);

  /// Power of ADScalar (e.g. \f$ (x_i)^a \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  pow(const ADScalar<ScalarF, number_of_derivativesF>& base,
      const ScalarF& power);

  /// Power of ADScalar (using \f$ (x)^y = exp(ln(x)y) \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  pow(const ADScalar<ScalarF, number_of_derivativesF>& base,
      const ADScalar<ScalarF, number_of_derivativesF>& power);

  /// Square root of ADScalar (e.g. \f$ \sqrt{x_i} \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  sqrt(const ADScalar<ScalarF, number_of_derivativesF>& radicand);

  /// Natural logarithm of ADScalar (e.g. \f$ \log{x_i} \f$ )
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  log(const ADScalar<ScalarF, number_of_derivativesF>& xi);

  /// Logarithm to base 10 of ADScalar (e.g. \f$ \log_{10}{x_i} \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  log10(const ADScalar<ScalarF, number_of_derivativesF>& a);

  /// Cosine function of ADScalar (e.g. \f$ \cos{x_i} \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  cos(const ADScalar<ScalarF, number_of_derivativesF>& a);

  /// Sine function of ADScalar (e.g. \f$ \sin{x_i} \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  sin(const ADScalar<ScalarF, number_of_derivativesF>& a);

  /// Tangent function of ADScalar (e.g. \f$ \tan{x_i} \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  tan(const ADScalar<ScalarF, number_of_derivativesF>& a);

  /// Inverse Cosine function of ADScalar (e.g. \f$ \acos{x_i} \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  acos(const ADScalar<ScalarF, number_of_derivativesF>& a);

  /// Inverse Sine function of ADScalar (e.g. \f$ \asin{x_i} \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  asin(const ADScalar<ScalarF, number_of_derivativesF>& a);

  /// Inverse Tangent function of ADScalar (e.g. \f$ \atan{x_i} \f$)
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr ADScalar<ScalarF, number_of_derivativesF>
  atan(const ADScalar<ScalarF, number_of_derivativesF>& a);

  /// Greater operator with a Scalar
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr bool
  operator>(const ScalarF& scalar,
            const ADScalar<ScalarF, number_of_derivativesF>& adt);

  /// Greater equal operator  with a Scalar
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr bool
  operator>=(const ScalarF& scalar,
             const ADScalar<ScalarF, number_of_derivativesF>& adt);

  /// Smaller operator with a Scalar
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr bool
  operator<(const ScalarF& scalar,
            const ADScalar<ScalarF, number_of_derivativesF>& adt);

  /// Smaller equal operator  with a Scalar
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr bool
  operator<=(const ScalarF& scalar,
             const ADScalar<ScalarF, number_of_derivativesF>& adt);

  /// Equal operator with a Scalar
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr bool
  operator==(const ScalarF& scalar,
             const ADScalar<ScalarF, number_of_derivativesF>& adt);

  /// Unequal operator  with a Scalar
  template<typename ScalarF, int number_of_derivativesF>
  friend constexpr bool
  operator!=(const ScalarF& scalar,
             const ADScalar<ScalarF, number_of_derivativesF>& adt);

  /** @} */ // End of Friend Injections

}; // end class ADScalar

#include "mimi/utils/ad.inl"

/// This is F-continugous layout, which is consistent with mfem.
/// again, Column-Major Matrix!
/// I am probably going to regret this, but
/// default is dynamic
template<int height = -1,
         int width = -1,
         typename ADType = mimi::utils::ADScalar<double, height * width>>
class ADMatrix {
public:
  using value_type = ADType;
  using ADType_ = ADType;
  using ContainerType_ =
      std::conditional_t<(height * width > 0) ? false : true,
                         mimi::utils::Data<value_type>,
                         std::array<value_type, number_of_derivatives>>;
  using IndexType_ = int;

protected:
  ContainerType_ data_;

public:
  static constexpr const int kHeight = height;
  static constexpr const int kWidth = width;
  static constexpr const int kSize = height * width;
  static constexpr const bool kDynamic = (kSize > 0) ? false : true;

  // these values are set dynamically
  int h_;
  int w_;
  int s_; //

  constexpr const int& Height() const {
    if constexpr (kDynamic) {
      return h_;
    } else {
      return kHeight;
    }
  }

  constexpr const int& Width() const {
    if constexpr (kDynamic) {
      return w_;
    } else {
      return kWidth;
    }
  }

  /// the notion of Size() is different here and in MFEM
  /// don't use this for template based implementation
  constexpr const int& Size() const {
    if constexpr (kDynamic) {
      return s_;
    } else {
      return kSize;
    }
  }

  constexpr bool IsDynamic const { return kDyanamic; }

  /// COLUMN MAJOR!
  constexpr Type_& operator()(const int& i, const int& j) {
    assert(j * Height() + i < Size());
    return data_[j * Height() + i];
  }

  /// COLUMN MAJOR!
  constexpr const Type_& operator()(const int& i, const int& j) const {
    assert(j * Height() + i < Size());
    return data_[j * Height() + i];
  }

  constexpr Type_& operator[](const int& i) {
    assert(i < Size());
    return data_[i];
  }

  constexpr const Type_& operator[](const int& i) const {
    assert(i < Size());
    return data_[i];
  }

  constexpr ContainerType_::value_type* DataReadWrite() { return data_.data(); }
  constexpr const ContainerType_::value_type* DataReadWrite() const {
    return data_.data();
  }

  /// to have some sort of general behavior, we allow non-dynamic types to call
  /// this that will assert
  constexpr void Reallocate(const int h, const int w, const int n_wrt = -1) {
    if constexpr (!kDynamic) {
      if (h != height || w != weight) {
        mimi::utils::PrintAndThrowError("h and w doesn't match static size");
      }
      return;
    }

    // save size values only if it differs from current size
    if (h != h_ || w != w_) {
      h_ = h;
      w_ = w;
      s_ = h * w;

      // reallocate data
      data_.Reallocate(s_);
    }

    // we only call reallocate if n_wrt is positive
    if (n_wrt > 0) {
      for (auto& d : data_) {
        // reallocate data's derivatives.
        if (d.DerivativeReadWrite().size() != n_wrt) {
          d.DerivativeReadWrite().Reallocate(n_wrt);
        }
      }
    }
  }

  // reset with fill value
  constexpr void Reset(const double value = 0.0) {
    for (IndexType_ i{}; i < Size(); ++i) {
      auto& i_data = data_[i];

      i_data.ValueReadWrite() = value;
      i_data.SetActiveComponent(i); // this calls fill()
    }
  }

  /// reset while copying values
  constexpr void Reset(const double* from_data) {
    for (IndexType_ i{}; i < Size(); ++i) {
      auto& i_data = data_[i];

      i_data.ValueReadWrite() = from_data[i];
      i_data.SetActiveComponent(i); // this calls fill()
    }
  }

  /// this is probably the most frequently used ctor
  constexpr ADMatrix() = default;

  // similar to shape based, but copies value
  constexpr ADMatrix(const mfem::DenseMatrix& m_mat) {
    if constexpr (kDynamic) {
      h_ = m_mat.Height();
      w_ = m_mat.Width();
      s_ = h_ * w_;
    } else {
      assert(height == m_mat.Height());
      assert(width == m_mat.Width());
    }

    // loop and copy
    const double* m_mat_data = m_mat.GetData();
    for (IndexType_ i{}; i < Size(); ++i) {
      auto& i_data = data_[i];

      if constexpr (kDynamic) {
        i_data.DerivativeReadWrite().Reallocate(Size());
      }
      i_data.ValueReadWrite() = m_mat_data[i];
      i_data.SetActiveComponent(i); // this calls fill()
    }
  }

  constexpr ADMatrix(const mfem::Vector& m_vec, const int h, const int w) {
    if constexpr (kDynamic) {
      h_ = h;
      w_ = w;
      s_ = h_ * w_;
    } else {
      assert(Size() == m_vec.Size());
    }
    // loop any copy
    const double* m_mat_data = m_vec.GetData();
    for (IndexType_ i{}; i < Size(); ++i) {
      auto& i_data = data_[i];

      if constexpr (kDynamic) {
        i_data.DerivativeReadWrite().Reallocate(Size());
      }
      i_data.ValueReadWrite() = m_mat_data[i];
      i_data.SetActiveComponent(i); // this calls fill()
    }
  }

  constexpr ADMatrix(const mfem::Vector& m_vec) {
    static_assert(!kDynamic, "Non dynamic matrix requires h, w");

    // loop any copy
    const double* m_mat_data = m_mat.GetData();
    for (IndexType_ i{}; i < Size(); ++i) {
      auto& i_data = data_[i];

      i_data.ValueReadWrite() = m_mat_data[i];
      i_data.SetActiveComponent(i); // this calls fill()
    }
  }

  constexpr ADType_ Det() const {
    MIMI_FUNC()

    // square
    assert(Width() == Height());
    if (Width() == 2) {
      auto back_slash = data_[0];
      back_slash *= data_[3];
      auto slash = data_[1];
      slash *= data_[2];

      back_slash -= slash;

      return bask_slash;
    } else {
      // this is from mfem - saves some ops
      //  const double *d = data;
      //  return
      //     d[0] * (d[4] * d[8] - d[5] * d[7]) +
      //     d[3] * (d[2] * d[7] - d[1] * d[8]) +
      //     d[6] * (d[1] * d[5] - d[2] * d[4]);

      // very first term is out;
      /* first */
      auto out = data_[4];
      out *= data_[8];
      auto tmp2 = data_[5];
      tmp2 *= data_[7];
      out -= tmp2;
      out *= data_[0];

      /* second */
      auto tmp1 = data_[2];
      tmp1 *= data_[7];
      tmp2 = data_[1];
      tmp2 *= data_[8];
      tmp1 -= tmp2;
      tmp1 *= data_[3];

      // first + second
      out += tmp1;

      /* third */
      tmp1 = data_[1];
      tmp1 *= data_[5];
      tmp2 = data_[2];
      tmp2 *= data_[4];
      tmp1 -= tmp2;
      tmp1 *= data_[6];

      out += tmp1;

      return out;
    }

    mimi::utils::PrintAndThrowError("Det() went wrong");
    return {}
  }
};

/* implements frequently used vector/matrix/linalg operations, follows
 * MFEM's function naming */

/// Initializes with sub vector
/// this resets derivatives and copies values.
template<typename MatrixType>
GetSubVector(const mfem::Vector& m_vec,
             const mfem::Array<int>& v_dof,
             MatrixType& ad_mat) {
  MIMI_FUNC()

  assert(v_dof.Size() == ad_mat.Size());

  const double* m_data = m_vec.GetData();
  auto* ad_data = ad_mat.DataReadWrite();

  int i {}
  for (const int& v : v_dof) {
    ad_data->ValueReadWrite() = m_data[v];
    ad_data->SetActiveComponent(i);

    ++i;
    ++ad_data;
  }
}

template<typename MatA, typename MatB, typename MatAtB>
void MultAtB(const MatA& a, const MatB& b, MatAtB& atb) {
  MIMI_FUNC()

  assert(a.Width() == atb.Height() && b.Width() == atb.Width()
         && a.Height() == b.Height());
  int const& a_h = a.Height();
  int const& a_w = a.Width();
  int const& b_w = b.Width();

  // create tmp
  for (int i{}; i < a_w; ++i) {
    for (int j{}; j < b_w; ++j) {
      // get column start
      const auto* a_col = &a(0, i);
      const auto* b_col = &b(0, j);
      // creat tmp
      auto tmp = (*a_col++) * (*b_col++);
      // iterate col and accumulate tmp
      for (int k{1}; k < a_h; ++k) {
        tmp += (*a_col++) * (*b_col++);
      }

      // set atb
      atb(i, j) = std::move(tmp);
    }
  }
}

template<typename MatA, typename DetType, typename MatInv>
void CalcInverse(const MatA& a, const DetType* a_det, MatInv& b) {
  MIMI_FUNC()

  // square
  assert(a.Width() == a.Height());
  // io size match
  assert(a.Height() == b.Height());
  assert(a.Width() == b.Width());

  int const& d = a.Width();

  if (d == 2) {
    b(0, 0) = a(1, 1);
    b(1, 1) = a(0, 0);
    b(0, 1) = -a(0, 1);
    b(1, 0) = -a(1, 0);

    if (a_det) {
      b *= 1.0 / (*a_det);
    } else {
      b *= 1.0 / a.Det();
    }
  } else if (d == 3) {
    b(0, 0) = a(1, 1) * a(2, 2) - a(1, 2) * a(2, 1); // ei-fh
    b(0, 1) = a(0, 2) * a(2, 1) - a(0, 1) * a(2, 2); // ch-bi
    b(0, 2) = a(0, 1) * a(1, 2) - a(0, 2) * a(1, 1); // bf-ce
    b(1, 0) = a(1, 2) * a(2, 0) - a(1, 0) * a(2, 2); // fg-di
    b(1, 1) = a(0, 0) * a(2, 2) - a(0, 2) * a(2, 0); // ai-cg
    b(1, 2) = a(0, 2) * a(1, 0) - a(0, 0) * a(1, 2); // cd-af
    b(2, 0) = a(1, 0) * a(2, 1) - a(1, 1) * a(2, 0); // dh-eg
    b(2, 1) = a(0, 1) * a(2, 0) - a(0, 0) * a(2, 1); // bg-ah
    b(2, 2) = a(0, 0) * a(1, 1) - a(0, 1) * a(1, 0); // ae-bd

    if (a_det) {
      b *= 1.0 / (*a_det);
    } else {
      b *= 1.0 / a.Det();
    }
  }

  mimi::utils::PrintAndThrowError("CalcInverse() went wrong.");
}

/// AB = A @ B
template<typename MatA, typename MatB, typename MatAB>
void Mult(const MatA& A, const MatB& B, MatAB& AB) {
  MIMI_FUNC()

  // size check
  assert(A.Width() == B.Height());
  assert(A.Height() == AB.Height());
  assert(B.Width() == AB.Width());

  for (int i{}; i < B.Width(); ++i) {
    for (int j{}; j < A.Height(); ++j) {
      const auto* b_col = &B(0, i);
      auto tmp = (*b_col++) * A(j, 0);
      for (int k{1}; k < A.Width(); ++k) {
        tmp += (*b_col++) * A(j, k);
      }
      aAB(j, i) = std::move(tmp);
    }
  }
}

template<typename ScalarA, typename MatA, typename MatB, typename Mat_aAB>
void AddMult_a(const ScalarA& a, const MatA& A, const MatB& B, Mat_aAB& aAB) {
  MIMI_FUNC()

  // size check
  assert(A.Width() == B.Height());
  assert(A.Height() == aAB.Height());
  assert(B.Width() == aAB.Width());

  for (int i{}; i < B.Width(); ++i) {
    for (int j{}; j < A.Height(); ++j) {
      const auto* b_col = &B(0, i);
      auto tmp = (*b_col++) * A(j, 0);
      for (int k{1}; k < A.Width(); ++k) {
        tmp += (*b_col++) * A(j, k);
      }
      tmp *= a;
      AB(j, i) += tmp;
    }
  }
}

template<typename ScalarA, typename MatA, typename MatB, typename Mat_aAB>
void AddMult_a_ABt(const ScalarA& a,
                   const MatA& A,
                   const MatB& B,
                   Mat_aAB& aABt) {
  MIMI_FUNC()

  // size check
  assert(A.Width() == B.Width());
  assert(A.Height() == aABt.Height());
  assert(B.Height() == aABt.Width());

  const int& a_h = A.Height();
  const int& b_h = B.Height();

  const auto* a_data = A.GetData();
  const auto* b_data = B.GetData();
  auto* aabt_begin = aABt.GetData();

  for (int i{}; i < A.Width(); ++i) {
    auto* aabt_data = aabt_begin;
    for (int j{}; j < b.Height(); ++j) {
      auto tmp = a * b_data[j];
      for (int k{}; k < A.Height(); ++k) {
        (*aabt_data++) += a_data[k] * tmp;
      }
    }
    a_data += a_h;
    b_data += b_h;
  }
}

} // namespace mimi::utils
