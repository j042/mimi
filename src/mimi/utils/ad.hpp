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
template<typename Scalar, std::size_t number_of_derivatives = 0>
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
      std::conditional_t<number_of_derivatives == 0,
                         mimi::utils::Data<Scalar_>,
                         std::array<Scalar_, number_of_derivatives>>;

  friend class ADVector;

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
  /// Default Copy Constructor
  constexpr ADScalar(const ADT_& rhs) = default;

  /// Default Move Constructor
  constexpr ADScalar(ADT_&& rhs) = default;

  /// Empty constructor which initializes all member variables to zero
  ADScalar() : v_{}, d_{} {}

  /// Scalar Constructor without Derivative
  template<std::size_t dummy = number_of_derivatives,
           typename std::enable_if_t<dummy == 0>* = nullptr>
  ADScalar(const Scalar_& value, const IndexingType_& n_derivatives)
      : v_{value} {
    d_.Reallocate(n_derivatives);
    d_.Fill(Scalar_{})
  } // d_{size} initializes to Scalar{}

  /// Scalar Constructor without Derivative
  template<std::size_t dummy = number_of_derivatives,
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
  template<std::size_t dummy = number_of_derivatives,
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
  template<std::size_t dummy = number_of_derivatives,
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
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend std::ostream&
  operator<<(std::ostream& os,
             const ADScalar<ScalarF, number_of_derivatives>& a);

  /// Addition
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  operator+(const ScalarF& a,
            const ADScalar<ScalarF, number_of_derivatives>& b);

  /// Subtraction
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  operator-(const ScalarF& a,
            const ADScalar<ScalarF, number_of_derivatives>& b);

  /// Multiplication
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  operator*(const ScalarF& a,
            const ADScalar<ScalarF, number_of_derivatives>& b);

  /// Division
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  operator/(const ScalarF& a,
            const ADScalar<ScalarF, number_of_derivatives>& b);

  /// Natural exponent of ADScalar (e.g. \f$ \exp{x_i} \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  exp(const ADScalar<ScalarF, number_of_derivatives>& exponent);

  /// Absolute Value
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  abs(const ADScalar<ScalarF, number_of_derivatives>& base);

  /// Power of ADScalar (e.g. \f$ (x_i)^a \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  pow(const ADScalar<ScalarF, number_of_derivatives>& base,
      const ScalarF& power);

  /// Power of ADScalar (using \f$ (x)^y = exp(ln(x)y) \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  pow(const ADScalar<ScalarF, number_of_derivatives>& base,
      const ADScalar<ScalarF, number_of_derivatives>& power);

  /// Square root of ADScalar (e.g. \f$ \sqrt{x_i} \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  sqrt(const ADScalar<ScalarF, number_of_derivatives>& radicand);

  /// Natural logarithm of ADScalar (e.g. \f$ \log{x_i} \f$ )
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  log(const ADScalar<ScalarF, number_of_derivatives>& xi);

  /// Logarithm to base 10 of ADScalar (e.g. \f$ \log_{10}{x_i} \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  log10(const ADScalar<ScalarF, number_of_derivatives>& a);

  /// Cosine function of ADScalar (e.g. \f$ \cos{x_i} \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  cos(const ADScalar<ScalarF, number_of_derivatives>& a);

  /// Sine function of ADScalar (e.g. \f$ \sin{x_i} \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  sin(const ADScalar<ScalarF, number_of_derivatives>& a);

  /// Tangent function of ADScalar (e.g. \f$ \tan{x_i} \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  tan(const ADScalar<ScalarF, number_of_derivatives>& a);

  /// Inverse Cosine function of ADScalar (e.g. \f$ \acos{x_i} \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  acos(const ADScalar<ScalarF, number_of_derivatives>& a);

  /// Inverse Sine function of ADScalar (e.g. \f$ \asin{x_i} \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  asin(const ADScalar<ScalarF, number_of_derivatives>& a);

  /// Inverse Tangent function of ADScalar (e.g. \f$ \atan{x_i} \f$)
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr ADScalar<ScalarF, number_of_derivatives>
  atan(const ADScalar<ScalarF, number_of_derivatives>& a);

  /// Greater operator with a Scalar
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr bool
  operator>(const ScalarF& scalar,
            const ADScalar<ScalarF, number_of_derivatives>& adt);

  /// Greater equal operator  with a Scalar
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr bool
  operator>=(const ScalarF& scalar,
             const ADScalar<ScalarF, number_of_derivatives>& adt);

  /// Smaller operator with a Scalar
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr bool
  operator<(const ScalarF& scalar,
            const ADScalar<ScalarF, number_of_derivatives>& adt);

  /// Smaller equal operator  with a Scalar
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr bool
  operator<=(const ScalarF& scalar,
             const ADScalar<ScalarF, number_of_derivatives>& adt);

  /// Equal operator with a Scalar
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr bool
  operator==(const ScalarF& scalar,
             const ADScalar<ScalarF, number_of_derivatives>& adt);

  /// Unequal operator  with a Scalar
  template<typename ScalarF, std::size_t number_of_derivatives>
  friend constexpr bool
  operator!=(const ScalarF& scalar,
             const ADScalar<ScalarF, number_of_derivatives>& adt);

  /** @} */ // End of Friend Injections

}; // end class ADScalar

#include "mimi/utils/ad.inl"

// now vector style wrapper expecially for FWDJAC
// full runtime
template<int dim = 0>
class ADVector : public mimi::utils::Data<ADScalar<double, dim>> {
public:
  using Base_ = mimi::utils::Data<ADScalar<double, dim>>;
  using IndexType_ = int;

  // size based
  ADVector(const int n) : Base_(n) {
    // loop each element and initialize
    for (IndexType_ i{}; i < n; ++i) {
      auto& i_data = data_[i];

      i_data.d_.Reallocate(n);
      i_data.SetActiveComponent(i); // this calls fill()
    }
  }

  // similar to size based, but copies value
  ADVector(const mfem::Vector& m_vec) {
    const int n = m_vec.Size();
    for (IndexType_ i{}; i < n; ++i) {
      auto& ad_data = data_[i];
      i_data.v_ = m_vec[i];
      i_data.d_.Reallocate(n);
      i_data.SetActiveComponent(i); // this calls fill()
    }
  }

  using Base_::opeartor[];
  using Base_::opeartor();
};

} // namespace mimi::utils
