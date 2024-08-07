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

// Addition
template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
ADScalar<Scalar, number_of_derivatives>::operator+(
    const ADScalar<Scalar, number_of_derivatives>& b) const {
  ADT_ result_value{(*this)};
  result_value += b;
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>&
ADScalar<Scalar, number_of_derivatives>::operator+=(
    const ADScalar<Scalar, number_of_derivatives>& b) {
  assert(b.GetNumberOfDerivatives() == GetNumberOfDerivatives());
  v_ += b.v_;
  for (IndexingType_ i{}; i < GetNumberOfDerivatives(); i++) {
    d_[i] += b.d_[i];
  }
  return (*this);
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
ADScalar<Scalar, number_of_derivatives>::operator+(const Scalar& b) const {
  ADT_ result_value{(*this)};
  result_value += b;
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>&
ADScalar<Scalar, number_of_derivatives>::operator+=(const Scalar& b) {
  v_ += b;
  return (*this);
}

// Substraction and negation
template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
ADScalar<Scalar, number_of_derivatives>::operator-() const {
  ADT_ result_value{(*this)};
  result_value.v_ = -result_value.v_;
  for (IndexingType_ i{}; i < GetNumberOfDerivatives(); i++) {
    result_value.d_[i] = -result_value.d_[i];
  }
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
ADScalar<Scalar, number_of_derivatives>::operator-(
    const ADScalar<Scalar, number_of_derivatives>& b) const {
  assert(b.GetNumberOfDerivatives() == GetNumberOfDerivatives());
  ADT_ result_value{(*this)};
  result_value -= b;
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>&
ADScalar<Scalar, number_of_derivatives>::operator-=(
    const ADScalar<Scalar, number_of_derivatives>& b) {
  v_ -= b.v_;
  for (IndexingType_ i{}; i < GetNumberOfDerivatives(); i++) {
    d_[i] -= b.d_[i];
  }
  return (*this);
}
template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
ADScalar<Scalar, number_of_derivatives>::operator-(const Scalar& b) const {
  ADT_ result_value{(*this)};
  result_value -= b;
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>&
ADScalar<Scalar, number_of_derivatives>::operator-=(const Scalar& b) {
  v_ -= b;
  return (*this);
}

// Multiplication
template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
ADScalar<Scalar, number_of_derivatives>::operator*(
    const ADScalar<Scalar, number_of_derivatives>& b) const {
  ADT_ result_value{(*this)};
  result_value *= b;
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>&
ADScalar<Scalar, number_of_derivatives>::operator*=(
    const ADScalar<Scalar, number_of_derivatives>& b) {
  assert(b.GetNumberOfDerivatives() == GetNumberOfDerivatives());
  for (IndexingType_ i{}; i < GetNumberOfDerivatives(); i++) {
    d_[i] = b.d_[i] * v_ + b.v_ * d_[i]; // f'(x) = u'*v +  u*v'
  }
  // Must be at the end otherwise overwrite value
  v_ *= b.v_;
  return (*this);
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
ADScalar<Scalar, number_of_derivatives>::operator*(const Scalar& b) const {
  ADT_ result_value{(*this)};
  result_value *= b;
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>&
ADScalar<Scalar, number_of_derivatives>::operator*=(const Scalar& b) {
  v_ *= b;
  for (IndexingType_ i{}; i < GetNumberOfDerivatives(); i++) {
    d_[i] *= b;
  }
  return (*this);
}

// Division
template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
ADScalar<Scalar, number_of_derivatives>::operator/(
    const ADScalar<Scalar, number_of_derivatives>& b) const {
  ADT_ result_value{(*this)};
  result_value /= b;
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>&
ADScalar<Scalar, number_of_derivatives>::operator/=(
    const ADScalar<Scalar, number_of_derivatives>& b) {
  assert(b.GetNumberOfDerivatives() == GetNumberOfDerivatives());
  const Scalar_ inverse_b_squared = 1.0 / (b.v_ * b.v_);
  for (IndexingType_ i{}; i < GetNumberOfDerivatives(); i++) {
    // f'(x) = (u'*v -  u*v') / v^2
    d_[i] = (b.v_ * d_[i] - b.d_[i] * v_) * inverse_b_squared;
  }
  v_ /= b.v_;
  return (*this);
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
ADScalar<Scalar, number_of_derivatives>::operator/(const Scalar& b) const {
  ADT_ result_value{(*this)};
  result_value /= b;
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>&
ADScalar<Scalar, number_of_derivatives>::operator/=(const Scalar& b) {
  const Scalar inverse_b{1 / b};
  v_ *= inverse_b;
  for (IndexingType_ i{}; i < GetNumberOfDerivatives(); i++) {
    d_[i] *= inverse_b;
  }
  return (*this);
}

//////////////////////////////
// Friend Functions
//////////////////////////////

template<typename Scalar, std::size_t number_of_derivatives>
std::ostream& operator<<(std::ostream& os,
                         const ADScalar<Scalar, number_of_derivatives>& a) {
  os << a.v_;
  return os;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
operator+(const Scalar& a, const ADScalar<Scalar, number_of_derivatives>& b) {
  return b + a;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
operator-(const Scalar& a, const ADScalar<Scalar, number_of_derivatives>& b) {
  return (-b) + a;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
operator*(const Scalar& a, const ADScalar<Scalar, number_of_derivatives>& b) {
  return b * a;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
operator/(const Scalar& a, const ADScalar<Scalar, number_of_derivatives>& b) {
  ADScalar<Scalar, number_of_derivatives> result_value{b};
  Scalar deriv = -a / (b.v_ * b.v_);
  result_value.v_ *= -deriv;
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < result_value.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= deriv;
  }

  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
exp(const ADScalar<Scalar, number_of_derivatives>& exponent) {
  Scalar tmp;
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    tmp = std::exp(exponent.v_);
  } else {
    tmp = exp(exponent.v_);
  }
  ADScalar<Scalar, number_of_derivatives> result_value{exponent};
  result_value.v_ = Scalar{1.};
  result_value *= tmp;
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
abs(const ADScalar<Scalar, number_of_derivatives>& base) {
  return base.v_ > 0.0 ? base : (-base);
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
pow(const ADScalar<Scalar, number_of_derivatives>& base, const Scalar& power) {
  Scalar tmp;
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    tmp = std::pow(base.v_, power - 1.0);
  } else {
    tmp = pow(base.v_, power - 1.0);
  }
  ADScalar<Scalar, number_of_derivatives> result_value{base};
  result_value.v_ *= tmp;
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < base.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= power * tmp;
  }
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
pow(const ADScalar<Scalar, number_of_derivatives>& base,
    const ADScalar<Scalar, number_of_derivatives>& power) {
  return exp(log(base) * power);
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
sqrt(const ADScalar<Scalar, number_of_derivatives>& radicand) {
  Scalar root;
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    root = std::sqrt(radicand.v_);
  } else {
    root = sqrt(radicand.v_);
  }
  const Scalar& half_inverse_root = 0.5 / root;
  ADScalar<Scalar, number_of_derivatives> result_value{radicand};
  result_value.v_ = root;
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < radicand.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= half_inverse_root;
  }
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
log(const ADScalar<Scalar, number_of_derivatives>& xi) {
  using namespace std;              // required for intrinsic types
  const Scalar& temp = 1.0 / xi.v_; // Multiplication is cheaper than division
  ADScalar<Scalar, number_of_derivatives> result_value{xi};
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    result_value.v_ = std::log(xi.v_);
  } else {
    result_value.v_ = log(xi.v_);
  }
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < xi.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= temp;
  }
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
log10(const ADScalar<Scalar, number_of_derivatives>& a) {
  using namespace std; // requ ired for intrinsic types
  const Scalar& temp = 1.0 / (a.v_ * std::log(10.));
  ADScalar<Scalar, number_of_derivatives> result_value{a};
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    result_value.v_ = std::log10(a.v_);
  } else {
    result_value.v_ = log10(a.v_);
  }
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < a.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= temp;
  }
  return result_value;
}

/////////
// Trigonometric functions

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
cos(const ADScalar<Scalar, number_of_derivatives>& rad) {
  ADScalar<Scalar, number_of_derivatives> result_value{rad};
  Scalar sin_of_angle;
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    sin_of_angle = -std::sin(rad.v_);
    result_value.v_ = std::cos(rad.v_);
  } else {
    sin_of_angle = -sin(rad.v_);
    result_value.v_ = cos(rad.v_);
  }
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < rad.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= sin_of_angle;
  }
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
sin(const ADScalar<Scalar, number_of_derivatives>& rad) {
  ADScalar<Scalar, number_of_derivatives> result_value{rad};
  Scalar cos_of_angle;
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    cos_of_angle = std::cos(rad.v_);
    result_value.v_ = std::sin(rad.v_);
  } else {
    cos_of_angle = cos(rad.v_);
    result_value.v_ = sin(rad.v_);
  }
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < rad.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= cos_of_angle;
  }
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
tan(const ADScalar<Scalar, number_of_derivatives>& rad) {
  ADScalar<Scalar, number_of_derivatives> result_value{rad};
  Scalar auxiliary_value;
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    auxiliary_value = 1. / std::cos(rad.v_);
    auxiliary_value *= auxiliary_value;
    result_value.v_ = std::tan(rad.v_);
  } else {
    auxiliary_value = 1. / cos(rad.v_);
    auxiliary_value *= auxiliary_value;
    result_value.v_ = tan(rad.v_);
  }
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < rad.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= auxiliary_value;
  }
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
acos(const ADScalar<Scalar, number_of_derivatives>& rad) {
  ADScalar<Scalar, number_of_derivatives> result_value{rad};
  Scalar auxiliary_value;
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    auxiliary_value = -1. / std::sqrt(1 - rad.v_ * rad.v_);
    result_value.v_ = std::acos(rad.v_);
  } else {
    auxiliary_value = -1. / sqrt(1 - rad.v_ * rad.v_);
    result_value.v_ = cos(rad.v_);
  }
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < rad.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= auxiliary_value;
  }
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
asin(const ADScalar<Scalar, number_of_derivatives>& rad) {
  ADScalar<Scalar, number_of_derivatives> result_value{rad};
  Scalar auxiliary_value;
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    auxiliary_value = 1. / std::sqrt(1 - rad.v_ * rad.v_);
    result_value.v_ = std::asin(rad.v_);
  } else {
    auxiliary_value = 1. / std::sqrt(1 - rad.v_ * rad.v_);
    result_value.v_ = asin(rad.v_);
  }
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < rad.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= auxiliary_value;
  }
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr ADScalar<Scalar, number_of_derivatives>
atan(const ADScalar<Scalar, number_of_derivatives>& rad) {
  ADScalar<Scalar, number_of_derivatives> result_value{rad};
  const Scalar auxiliary_value{1. / (1. + rad.v_ * rad.v_)};
  if constexpr (std::is_arithmetic_v<Scalar>) {
    // Use STD namespace function
    result_value.v_ = std::atan(rad.v_);
  } else {
    result_value.v_ = atan(rad.v_);
  }
  for (typename ADScalar<Scalar, number_of_derivatives>::IndexingType_ i{};
       i < rad.GetNumberOfDerivatives();
       i++) {
    result_value.d_[i] *= auxiliary_value;
  }
  return result_value;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr bool operator>(const Scalar& scalar,
                         const ADScalar<Scalar, number_of_derivatives>& adt) {
  return adt < scalar;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr bool operator>=(const Scalar& scalar,
                          const ADScalar<Scalar, number_of_derivatives>& adt) {
  return adt <= scalar;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr bool operator<(const Scalar& scalar,
                         const ADScalar<Scalar, number_of_derivatives>& adt) {
  return adt > scalar;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr bool operator<=(const Scalar& scalar,
                          const ADScalar<Scalar, number_of_derivatives>& adt) {
  return adt >= scalar;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr bool operator==(const Scalar& scalar,
                          const ADScalar<Scalar, number_of_derivatives>& adt) {
  return adt == scalar;
}

template<typename Scalar, std::size_t number_of_derivatives>
constexpr bool operator!=(const Scalar& scalar,
                          const ADScalar<Scalar, number_of_derivatives>& adt) {
  return !(adt == scalar);
}
