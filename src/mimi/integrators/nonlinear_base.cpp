#include <cmath>

#include "mimi/integrators/nonlinear_base.hpp"

namespace mimi::integrators {

void NonlinearBase::Prepare() {
  MIMI_FUNC()

  mimi::utils::PrintAndThrowError("Prepare not implemented");
}

void NonlinearBase::AddDomainResidual(const mfem::Vector& current_x,
                                      mfem::Vector& residual) const {
  mimi::utils::PrintAndThrowError("AddDomainResidual not implemented");
}

void NonlinearBase::AddDomainGrad(const mfem::Vector& current_x,
                                  mfem::SparseMatrix& grad) const {
  mimi::utils::PrintAndThrowError("AddDomainGrad not implemented");
}

void NonlinearBase::AddDomainResidualAndGrad(const mfem::Vector& current_x,
                                             const double grad_factor,
                                             mfem::Vector& residual,
                                             mfem::SparseMatrix& grad) const {
  mimi::utils::PrintAndThrowError("AddDomainResidualAndGrad not implemented");
}

void NonlinearBase::DomainPostTimeAdvance(const mfem::Vector& current_x) {
  mimi::utils::PrintAndThrowError("DomainPostTimeAdvance not implemented");
}

void NonlinearBase::AddBoundaryResidual(const mfem::Vector& current_x,
                                        mfem::Vector& residual) {
  mimi::utils::PrintAndThrowError("AddBoundaryResidual not implemented");
}

void NonlinearBase::AddBoundaryGrad(const mfem::Vector& current_x,
                                    mfem::SparseMatrix& grad) {
  mimi::utils::PrintAndThrowError("AddBoundaryGrad not implemented");
}

void NonlinearBase::AddBoundaryResidualAndGrad(const mfem::Vector& current_x,
                                               const double grad_factor,
                                               mfem::Vector& residual,
                                               mfem::SparseMatrix& grad) {
  mimi::utils::PrintAndThrowError("AddBoundaryResidualAndGrad not implemented");
}

void NonlinearBase::BoundaryPostTimeAdvance(const mfem::Vector& current_x) {
  MIMI_FUNC()
  mimi::utils::PrintAndThrowError("BoundaryPostTimeAdvance not implemented");
}

mimi::utils::Vector<int>& NonlinearBase::MarkedBoundaryElements() {
  MIMI_FUNC()
  if (!boundary_marker_) {
    mimi::utils::PrintAndThrowError("boundary_marker_ not set");
  }
  if (!precomputed_) {
    mimi::utils::PrintAndThrowError("precomputed_ not set");
  }

  if (marked_boundary_elements_.size() > 0) {
    return marked_boundary_elements_;
  }

  auto& mesh = *precomputed_->meshes_[0];
  auto& b_marker = *boundary_marker_;
  marked_boundary_elements_.reserve(precomputed_->n_b_elements_);

  // loop boundary elements and check their attributes
  for (int i{}; i < precomputed_->n_b_elements_; ++i) {
    const int bdr_attr = mesh.GetBdrAttribute(i);
    if (b_marker[bdr_attr - 1] == 0) {
      continue;
    }
    marked_boundary_elements_.push_back(i);
  }
  marked_boundary_elements_.shrink_to_fit();

  return marked_boundary_elements_;
}

double NonlinearBase::GapNorm(const mfem::Vector& test_x, const int nthreads) {
  MIMI_FUNC()
  mimi::utils::PrintAndThrowError("GapNorm(x, nthread) not implemented for",
                                  Name());
  return 0.0;
}

} // namespace mimi::integrators
