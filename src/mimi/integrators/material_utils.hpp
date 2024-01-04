#pragma once

#include <algorithm>
#include <cmath>

#include <mfem.hpp>

#include "mimi/utils/print.hpp"

namespace mimi::integrators {

/// computes deviator.
/// often used to compute stress deviator, so there's factor
inline void Dev(const mfem::DenseMatrix& A,
                const int dim,
                const double factor,
                mfem::DenseMatrix& dev_A) {
  // get data ptr
  const double* A_data = A.GetData();
  double* dev_A_data = dev_A.GetData();

  if (dim == 2) {
    const double& A_0 = A_data[0];
    const double& A_3 = A_data[3];
    const double tr_A_over_dim = (A_0 + A_3) / 2.0; // div(val, dim)

    dev_A_data[0] = (A_0 - tr_A_over_dim) * factor;
    dev_A_data[1] = A_data[1] * factor;
    dev_A_data[2] = A_data[2] * factor;
    dev_A_data[3] = (A_3 - tr_A_over_dim) * factor;
    return;
  } else {
    const double& A_0 = A_data[0];
    const double& A_4 = A_data[4];
    const double& A_8 = A_data[8];
    const double tr_A_over_dim = (A_0 + A_4 + A_8) / 3.0; // div(val, dim)

    dev_A_data[0] = (A_0 - tr_A_over_dim) * factor;
    dev_A_data[1] = A_data[1] * factor;
    dev_A_data[2] = A_data[2] * factor;
    dev_A_data[3] = A_data[3] * factor;
    dev_A_data[4] = (A_4 - tr_A_over_dim) * factor;
    dev_A_data[5] = A_data[5] * factor;
    dev_A_data[6] = A_data[6] * factor;
    dev_A_data[7] = A_data[7] * factor;
    dev_A_data[8] = (A_8 - tr_A_over_dim) * factor;
    return;
  }
}

/// sym(F - I) - plastic strain
/// or , sym(F) - I - plastic strain
inline void ElasticStrain(const mfem::DenseMatrix& F,
                          const mfem::DenseMatrix& plastic_strain,
                          mfem::DenseMatrix& elastic_strain) {
  const int dim = F.Height();
  const double* F_data = F.GetData();
  const double* p_data = plastic_strain.GetData();
  double* e_data = elastic_strain.GetData();

  // first, use space of e to copy F
  std::copy_n(F_data, dim * dim, e_data);

  // symmetrize
  elastic_strain.Symmetrize();

  // substract I
  for (int i{}; i < dim; ++i) {
    e_data[(dim + 1) * i] -= 1;
  }

  // subtract plastic_strain
  for (int i{}; i < dim * dim; ++i) {
    e_data[i] -= p_data[i];
  }
}

/// Frobenius norm of a matrix
inline double Norm(const mfem::DenseMatrix& A) {
  const int dim = A.Height();

  const double* A_data = A.GetData();
  double a{};
  for (int i{}; i < dim * dim; ++i) {
    const double& A_i = A_data[i];
    a += A_i * A_i;
  }
  return std::sqrt(a);
}

/// A += a * I
inline void AddDiag(mfem::DenseMatrix& A, const double a) {
  const int dim = A.Height();
  double* A_data = A.GetData();
  for (int i{}; i < dim; ++i) {
    A_data[(dim + 1) * i] += a;
  }
}

/// computes L = F_dot * F_inv
inline void VelocityGradient(const mfem::DenseMatrix& F,
                             const mfem::DenseMatrix& F_dot,
                             mfem::DenseMatrix& L) {
  MIMI_FUNC()

  double F_inv_data[9];
  const int dim = F_dot.Width();
  mfem::DenseMatrix F_inv(F_inv_data, dim, dim);
  mfem::CalcInverse(F, F_inv);

  mfem::Mult(F_dot, F_inv, L);
}

/// @brief evaluates equivalent strain rate based on
/// https://www.sciencedirect.com/science/article/pii/S1359645411002412
/// https://hal.science/hal-03244237/document
/// ... is this actually same as normal one?
/// @param F_dot
/// @return
inline double EquivalentPlasticStrainRate2D(const mfem::DenseMatrix& F_dot) {
  MIMI_FUNC()

  const double* fd = F_dot.GetData();
  // Those are just elements of sym(F_dot)
  const double e_xx = fd[0];
  const double e_yy = fd[3];
  // gamma = 2 * e_xy | e_xy = 0.5 * f1 * f2
  const double gamma = fd[1] + fd[2];

  // sqrt( 4/9 * (1/2 * ( (e_xx - e_yy)^2 + e_xx^2 + e_yy^2 ) + 3/4*gamma^2 )
  // sqrt( 2/9 * (e_xx - e_yy)^2 + e_xx^2 + e_yy^2) + 1/3 * gamma^2
  return std::sqrt(
      2. / 9. * ((e_xx - e_yy) * (e_xx - e_yy) + e_xx * e_xx + e_yy * e_yy)
      + 1. / 3. * gamma * gamma);
}

/// @brief sqrt(2/3  s_dot_ij  s_dot_ij)
/// Temporary, maybe removed
/// @param F_dot
/// @return
inline double EquivalentPlasticStrainRate_Dev(const mfem::DenseMatrix& F_dot) {
  MIMI_FUNC()

  const double* f = F_dot.GetData();
  const int dim = F_dot.Width();

  // steps
  // 1. devL = dev(sym(F_dot))
  // 2. sqrt(2/3 devL : devL)

  if (dim == 2) {
    const double trace_over_dim = (f[0] + f[3]) / 2.0;

    const double d0 = f[0] - trace_over_dim;
    const double d1 = 0.5 * f[1] * f[2];
    const double d3 = f[3] - trace_over_dim;

    return std::sqrt(2. / 3. * (d0 * d0 + 2.0 * d1 * d1 + d3 * d3));
  } else {
    const double trace_over_dim = (f[0] + f[4] + f[8]) / 3.0;

    const double d0 = f[0] - trace_over_dim;

    const double d1 = 0.5 * (f[1] + f[3]);
    const double d2 = 0.5 * (f[2] + f[6]);

    const double d4 = f[4] - trace_over_dim;

    const double d5 = 0.5 * (f[5] + f[7]);

    const double d8 = f[8] - trace_over_dim;

    return std::sqrt(2. / 3.
                     * (d0 * d0 + 2. * d1 * d1 + 2. * d2 * d2 + d4 * d4
                        + 2. * d5 * d5 + d8 * d8));
  }

  return {};
}

/// @brief sqrt(2/3  eps_dot_ij  eps_dot_ij)
/// Temporary, maybe removed
/// @param F_dot
/// @return
inline double EquivalentPlasticStrainRate(const mfem::DenseMatrix& F_dot) {
  MIMI_FUNC()

  const double* f = F_dot.GetData();
  const int dim = F_dot.Width();

  // steps
  // 1. devL = dev(sym(F_dot))
  // 2. sqrt(2/3 devL : devL)

  if (dim == 2) {
    const double d0 = f[0];
    const double d1 = 0.5 * f[1] * f[2];
    const double d3 = f[3];

    return std::sqrt(2. / 3. * (d0 * d0 + 2.0 * d1 * d1 + d3 * d3));
  } else {
    const double d0 = f[0];

    const double d1 = 0.5 * (f[1] + f[3]);
    const double d2 = 0.5 * (f[2] + f[6]);

    const double d4 = f[4];

    const double d5 = 0.5 * (f[5] + f[7]);

    const double d8 = f[8];

    return std::sqrt(2. / 3.
                     * (d0 * d0 + 2. * d1 * d1 + 2. * d2 * d2 + d4 * d4
                        + 2. * d5 * d5 + d8 * d8));
  }

  return {};
}

} // namespace mimi::integrators
