#pragma once

#include <mfem.hpp>

#include "mimi/utils/containers.hpp"
#include "mimi/utils/precomputed.hpp"
#include "mimi/coefficients/nearest_distance.hpp"

namespace mimi::integrators {

/// Temporary containers required in element assembly. One for each thread will
/// be created. Stores values required for material, quad point and element
/// data.
class NonlinearSolidWorkData {
public:
  // basic info. used to compute tdofs
  int dim_;
  int n_dof_;

  // state variable
  double det_F_{};
  bool has_det_F_{false};
  bool has_F_inv_{false};

  /// flag to inform residual destination
  bool assembling_grad_{false};

  // general assembly items
  mfem::Vector element_x_;
  mfem::DenseMatrix element_x_mat_;
  mfem::DenseMatrix local_residual_;
  mfem::DenseMatrix local_grad_;
  mfem::DenseMatrix forward_residual_;

  // used for materials
  mfem::DenseMatrix stress_;
  mfem::DenseMatrix F_;
  mfem::DenseMatrix F_inv_;
  mfem::DenseMatrix F_dot_; // for visco but put it here for easier visibility

  // used in materials - materials will visit and initiate those in
  // PrepareWorkData
  mfem::DenseMatrix I_;
  mfem::DenseMatrix alternative_stress_;           // for conversion
  mimi::utils::Vector<mfem::Vector> aux_vec_;      // for computation
  mimi::utils::Vector<mfem::DenseMatrix> aux_mat_; // for computation

  // this is global sized local residual / grad entries
  mfem::Vector thread_local_residual_;
  mfem::Vector thread_local_A_;

  /// this can be called once at Prepare()
  void SetDim(const int dim);

  /// this should be called at the start of every element as NDof may change
  void SetDof(const int n_dof);

  /// Number of true dof based on dim and n_dof values
  int GetTDof() const { return dim_ * n_dof_; }

  /// computes F, deformation gradient, and resets flags
  void ComputeF(const mfem::DenseMatrix& dNdX);

  /// Returns inverse of F. For consecutive calls, it will return stored value
  mfem::DenseMatrix& FInv() {
    MIMI_FUNC()
    if (has_F_inv_) {
      return F_inv_;
    }
    mfem::CalcInverse(F_, F_inv_);

    has_F_inv_ = true;
    return F_inv_;
  }

  /// determinant of F
  double DetF() {
    MIMI_FUNC()
    if (has_det_F_) {
      return det_F_;
    }

    det_F_ = F_.Det();
    has_det_F_ = true;
    return det_F_;
  }

  /// Given global state vector and support ids, copies element vector and
  /// returns matrix form.
  mfem::DenseMatrix& CurrentElementSolutionCopy(const mfem::Vector& current_all,
                                                const mfem::Array<int>& vdofs);

  mfem::DenseMatrix& CurrentSolution() {
    MIMI_FUNC()

    return element_x_mat_;
  }

  /// hint flag to inform current action. influences return value for
  /// ResidualMatrix()
  void GradAssembly(bool state) {
    MIMI_FUNC()

    assembling_grad_ = state;
  }

  /// Returns destination matrix for assembly. This maybe for residual or FD.
  mfem::DenseMatrix& ResidualMatrix() {
    MIMI_FUNC()
    if (assembling_grad_) {
      return forward_residual_;
    }
    return local_residual_;
  }
};

/// temporary containers required in element assembly
/// mfem performs some fancy checks for allocating memories.
/// So we create one for each thread
class MortarContactWorkData {
public:
  using ElementQuadData_ = mimi::utils::ElementQuadData;

  // array indices for look up
  static const int kDof{0};
  static const int kVDof{1};

  static const int kXRef{0};

  mfem::Vector element_x_;
  mfem::DenseMatrix element_x_mat_;
  mfem::DenseMatrix J_;
  mfem::DenseMatrix local_residual_;
  mfem::DenseMatrix local_grad_;
  mfem::DenseMatrix forward_residual_;

  mfem::Vector element_pressure_;

  mimi::coefficients::NearestDistanceBase::Query distance_query_;
  mimi::coefficients::NearestDistanceBase::Results distance_results_;

  mfem::Vector normal_;

  // thread local
  mfem::Vector local_area_;
  mfem::Vector local_gap_;
  // global, read as thread_local (pause) area
  mfem::Vector thread_local_area_;
  mfem::Vector thread_local_gap_;

  int dim_{-1};
  int n_dof_{-1};
  bool assembling_grad_{false};

  void SetDim(const int dim);

  void SetDof(const int n_dof);

  mfem::DenseMatrix&
  CurrentElementSolutionCopy(const mfem::Vector& current_all,
                             const ElementQuadData_& eq_data);

  mfem::Vector& CurrentElementPressure(const mfem::Vector& pressure_all,
                                       const ElementQuadData_& eq_data);

  mfem::DenseMatrix& ComputeJ(const mfem::DenseMatrix& dndxi);

  void ComputeNearestDistanceQuery(const mfem::Vector& N);

  bool IsPressureZero() const;

  double Pressure(const mfem::Vector& N) const;

  void InitializeThreadLocalAreaAndGap();

  void GradAssembly(bool state) {
    MIMI_FUNC()

    assembling_grad_ = state;
  }

  mfem::DenseMatrix& ResidualMatrix() {
    MIMI_FUNC()
    if (assembling_grad_) {
      return forward_residual_;
    }
    return local_residual_;
  }

  int GetTDof() const { return dim_ * n_dof_; }

  mfem::DenseMatrix& CurrentSolution() {
    MIMI_FUNC()

    return element_x_mat_;
  }
};

/// simple rewrite of mfem::AddMult_a_VWt()
template<typename DataType>
void Ptr_AddMult_a_VWt(const DataType a,
                       const DataType* v_begin,
                       const DataType* v_end,
                       const DataType* w_begin,
                       const DataType* w_end,
                       DataType* aVWt) {
  DataType* out{aVWt};
  for (const DataType* wi{w_begin}; wi != w_end; ++wi) {
    const DataType aw = *wi * a;
    for (const DataType* vi{v_begin}; vi != v_end;) {
      (*out++) += aw * (*vi++);
    }
  }
}

template<typename DerivativeContainer, typename NormalContainer>
static inline void
ComputeUnitNormal(const DerivativeContainer& first_derivatives,
                  NormalContainer& unit_normal) {
  MIMI_FUNC()

  const int dim = unit_normal.Size();
  const double* jac_d = first_derivatives.GetData();
  double* unit_normal_d = unit_normal.GetData();
  if (dim == 2) {
    const double d0 = jac_d[0];
    const double d1 = jac_d[1];
    const double inv_norm2 = 1. / std::sqrt(d0 * d0 + d1 * d1);
    unit_normal_d[0] = d1 * inv_norm2;
    unit_normal_d[1] = -d0 * inv_norm2;

    // this should be either 2d or 3d
  } else {
    const double d0 = jac_d[0];
    const double d1 = jac_d[1];
    const double d2 = jac_d[2];
    const double d3 = jac_d[3];
    const double d4 = jac_d[4];
    const double d5 = jac_d[5];

    const double n0 = d1 * d5 - d2 * d4;
    const double n1 = d2 * d3 - d0 * d5;
    const double n2 = d0 * d4 - d1 * d3;

    const double inv_norm2 = 1. / std::sqrt(n0 * n0 + n1 * n1 + n2 * n2);

    unit_normal_d[0] = n0 * inv_norm2;
    unit_normal_d[1] = n1 * inv_norm2;
    unit_normal_d[2] = n2 * inv_norm2;
  }
}

} // namespace mimi::integrators