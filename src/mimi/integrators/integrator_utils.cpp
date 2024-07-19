#include "mimi/integrators/integrator_utils.hpp"

namespace mimi::integrators {

void NonlinearSolidWorkData::SetDim(const int dim) {
  MIMI_FUNC()
  has_det_F_ = false;
  has_F_inv_ = false;

  dim_ = dim;

  stress_.SetSize(dim, dim);
  F_.SetSize(dim, dim);
  F_inv_.SetSize(dim, dim);
  F_dot_.SetSize(dim, dim);

  I_.Diag(1., dim);
  alternative_stress_.SetSize(dim, dim);
}

/// this should be called at the start of every element as NDof may change
void NonlinearSolidWorkData::SetDof(const int n_dof) {
  MIMI_FUNC()

  n_dof_ = n_dof;
  element_x_.SetSize(n_dof * dim_); // will be resized in getsubvector
  element_x_mat_.UseExternalData(element_x_.GetData(), n_dof, dim_);
  forward_residual_.SetSize(n_dof, dim_);
  local_residual_.SetSize(n_dof, dim_);
  local_grad_.SetSize(n_dof * dim_, n_dof * dim_);
}

void NonlinearSolidWorkData::ComputeF(const mfem::DenseMatrix& dNdX) {
  MIMI_FUNC()
  has_det_F_ = false;
  has_F_inv_ = false;

  mfem::MultAtB(element_x_mat_, dNdX, F_);
  mimi::utils::AddDiagonal(F_.GetData(), 1.0, dim_);
}

/// Given global state vector and support ids, copies element vector and
/// returns matrix form.
mfem::DenseMatrix& NonlinearSolidWorkData::CurrentElementSolutionCopy(
    const mfem::Vector& current_all,
    const mfem::Array<int>& vdofs) {
  MIMI_FUNC()
  current_all.GetSubVector(vdofs, element_x_);

  return element_x_mat_;
}

void MortarContactWorkData::SetDim(const int dim) {
  MIMI_FUNC()

  dim_ = dim;
  assert(dim_ > 0);

  distance_query_.SetSize(dim);
  distance_results_.SetSize(dim - 1, dim);
  J_.SetSize(dim_, dim_ - 1);
  normal_.SetSize(dim_);
}

void MortarContactWorkData::SetDof(const int n_dof) {
  MIMI_FUNC()

  assert(dim_ > 0);
  n_dof_ = n_dof;
  element_x_.SetSize(n_dof * dim_);
  element_x_mat_.UseExternalData(element_x_.GetData(), n_dof, dim_);
  local_residual_.SetSize(n_dof, dim_);
  local_grad_.SetSize(n_dof * dim_, n_dof * dim_);
  forward_residual_.SetSize(n_dof, dim_);
  element_pressure_.SetSize(n_dof);
  local_area_.SetSize(n_dof);
  local_gap_.SetSize(n_dof);
}

mfem::DenseMatrix& MortarContactWorkData::CurrentElementSolutionCopy(
    const mfem::Vector& current_all,
    const ElementQuadData_& eq_data) {
  MIMI_FUNC()

  current_all.GetSubVector(eq_data.GetElementData().v_dofs, element_x_);
  element_x_mat_ += eq_data.GetMatrix(kXRef);

  return element_x_mat_;
}

mfem::Vector&
MortarContactWorkData::CurrentElementPressure(const mfem::Vector& pressure_all,
                                              const ElementQuadData_& eq_data) {
  MIMI_FUNC()

  pressure_all.GetSubVector(eq_data.GetArray(kDof), element_pressure_);
  return element_pressure_;
}

mfem::DenseMatrix&
MortarContactWorkData::ComputeJ(const mfem::DenseMatrix& dndxi) {
  MIMI_FUNC()
  mfem::MultAtB(element_x_mat_, dndxi, J_);
  return J_;
}

void MortarContactWorkData::ComputeNearestDistanceQuery(const mfem::Vector& N) {
  MIMI_FUNC()
  element_x_mat_.MultTranspose(N, distance_query_.query.data());
}

bool MortarContactWorkData::IsPressureZero() const {
  for (const double p : element_pressure_) {
    if (p != 0.0) {
      return false;
    }
  }
  return true;
}

double MortarContactWorkData::Pressure(const mfem::Vector& N) const {
  MIMI_FUNC()
  const double* p_d = element_pressure_.GetData();
  const double* n_d = N.GetData();
  double p{};
  for (int i{}; i < N.Size(); ++i) {
    p += n_d[i] * p_d[i];
  }
  return p;
}

void MortarContactWorkData::InitializeThreadLocalAreaAndGap() {
  double* a_d = thread_local_area_.GetData();
  double* g_d = thread_local_gap_.GetData();
  for (int i{}; i < thread_local_area_.Size(); ++i) {
    a_d[i] = 0.0;
    g_d[i] = 0.0;
  }
}

} // namespace mimi::integrators
