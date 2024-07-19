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

} // namespace mimi::integrators
