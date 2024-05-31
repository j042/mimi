#pragma once

#include <mfem.hpp>

#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::forms {
/* Extends Mult() to incorporate mimi::integrators::nonloinear_base */
class Nonlinear : public mfem::NonlinearForm {
public:
  using Base_ = mfem::NonlinearForm;
  using NFIPointer_ = std::shared_ptr<mimi::integrators::NonlinearBase>;

  mimi::utils::Vector<NFIPointer_> domain_nfi_{};
  mimi::utils::Vector<NFIPointer_> boundary_face_nfi_{};
  mimi::utils::Vector<const mfem::Array<int>*> boundary_markers_{};
  std::shared_ptr<const bool> operator_frozen_state_;

  virtual void SetAndPassOperatorFrozenState(
      std::shared_ptr<const bool> operator_frozen_state) {
    MIMI_FUNC()
    operator_frozen_state_ = operator_frozen_state;
    for (auto& dnfi : domain_nfi_) {
      dnfi->operator_frozen_state_ = operator_frozen_state_;
    }
    for (auto& bfnfi : boundary_face_nfi_) {
      bfnfi->operator_frozen_state_ = operator_frozen_state_;
    }
  }

  using Base_::Base_;

  /// time step size, in case you need them
  /// operators should set them
  double dt_{0.0};
  double first_effective_dt_{0.0};  // this is for x
  double second_effective_dt_{0.0}; // this is for x_dot (=v).

  virtual void AssembleGradOn() {
    MIMI_FUNC()
    for (auto& dnfi : domain_nfi_) {
      dnfi->assemble_grad_ = true;
    }
    for (auto& bnfi : boundary_face_nfi_) {
      bnfi->assemble_grad_ = true;
    }
  }

  virtual void AssembleGradOff() {
    MIMI_FUNC()
    for (auto& dnfi : domain_nfi_) {
      dnfi->assemble_grad_ = false;
    }
    for (auto& bnfi : boundary_face_nfi_) {
      bnfi->assemble_grad_ = false;
    }
  }

  /// we skip a lot of checks that's performed by base here
  virtual void Mult(const mfem::Vector& current_x,
                    mfem::Vector& residual) const {
    MIMI_FUNC()

    residual = 0.0;

    AddMult(current_x, residual);
  }

  /// this is not an override.
  /// This does not initialize residual with 0. instead just adds to it.
  /// Mult() uses this after initializing residual to zero
  virtual void AddMult(const mfem::Vector& current_x,
                       mfem::Vector& residual) const {
    MIMI_FUNC()

    // we assemble all first - these will call nthreadexe
    // domain
    for (auto& domain_integ : domain_nfi_) {
      domain_integ->dt_ = dt_;
      domain_integ->first_effective_dt_ = first_effective_dt_;
      domain_integ->second_effective_dt_ = second_effective_dt_;

      domain_integ->AddDomainResidual(current_x, -1, residual);
    }

    // boundary
    for (auto& boundary_integ : boundary_face_nfi_) {
      boundary_integ->dt_ = dt_;
      boundary_integ->first_effective_dt_ = first_effective_dt_;
      boundary_integ->second_effective_dt_ = second_effective_dt_;
      boundary_integ->AddBoundaryResidual(current_x, -1, residual);
    }

    // set true dofs - if we have time, we could use nthread this.
    for (const auto& tdof : Base_::ess_tdof_list) {
      residual[tdof] = 0.0;
    }
  }

  virtual void AddResidual(const mfem::Vector& current_x,
                           const int nthread,
                           mfem::Vector& residual) const {
    MIMI_FUNC()

    // we assemble all first - these will call nthreadexe
    // domain
    for (auto& domain_integ : domain_nfi_) {
      domain_integ->dt_ = dt_;
      domain_integ->first_effective_dt_ = first_effective_dt_;
      domain_integ->second_effective_dt_ = second_effective_dt_;
      domain_integ->AddDomainResidual(current_x, nthread, residual);
    }

    // boundary
    for (auto& boundary_integ : boundary_face_nfi_) {
      boundary_integ->dt_ = dt_;
      boundary_integ->first_effective_dt_ = first_effective_dt_;
      boundary_integ->second_effective_dt_ = second_effective_dt_;
      boundary_integ->AddBoundaryResidual(current_x, nthread, residual);
    }

    // set true dofs - if we have time, we could use nthread this.
    for (const auto& tdof : Base_::ess_tdof_list) {
      residual[tdof] = 0.0;
    }
  }

  virtual void AddMultGrad(const mfem::Vector& current_x,
                           const int nthread,
                           const double grad_factor,
                           mfem::Vector& residual,
                           mfem::SparseMatrix& grad) const {
    for (auto& domain_integ : domain_nfi_) {
      domain_integ->dt_ = dt_;
      domain_integ->first_effective_dt_ = first_effective_dt_;
      domain_integ->second_effective_dt_ = second_effective_dt_;
      domain_integ->AddDomainResidualAndGrad(current_x,
                                             nthread,
                                             grad_factor,
                                             residual,
                                             grad);
    }

    // boundary
    for (auto& boundary_integ : boundary_face_nfi_) {
      boundary_integ->dt_ = dt_;
      boundary_integ->first_effective_dt_ = first_effective_dt_;
      boundary_integ->second_effective_dt_ = second_effective_dt_;
      boundary_integ->AddBoundaryResidualAndGrad(current_x,
                                                 nthread,
                                                 grad_factor,
                                                 residual,
                                                 grad);
    }

    // set true dofs - if we have time, we could use nthread this.
    for (const auto& tdof : Base_::ess_tdof_list) {
      residual[tdof] = 0.0;
      grad.EliminateRowCol(tdof);
    }
  }

  virtual mfem::Operator& GetGradient(const mfem::Vector& current_x) const {
    MIMI_FUNC();

    if (Grad == NULL) {
      // this is an adhoc solution to get sparsity pattern
      // we know one of nl integrator should have precomputed, so access matrix
      // from that
      mfem::SparseMatrix* sparsity_pattern;
      if (domain_nfi_.size() > 0) {
        sparsity_pattern =
            domain_nfi_[0]->precomputed_->sparsity_pattern_.get();
      } else if (boundary_face_nfi_.size() > 0) {
        sparsity_pattern =
            boundary_face_nfi_[0]->precomputed_->sparsity_pattern_.get();
      } else {
        mimi::utils::PrintAndThrowError(
            "No sparsity pattern from precomputed.");
      }
      if (!sparsity_pattern) {
        mimi::utils::PrintAndThrowError("Invalid sparsity pattern saved");
      }

      Base_::Grad = new mfem::SparseMatrix(*sparsity_pattern); // deep copy
      *Base_::Grad = 0.0;
    } else {
      *Base_::Grad = 0.0;
    }

    // we assemble all first - these will call nthreadexe
    // domain
    for (auto& domain_integ : domain_nfi_) {
      domain_integ->dt_ = dt_;
      domain_integ->first_effective_dt_ = first_effective_dt_;
      domain_integ->second_effective_dt_ = second_effective_dt_;
      domain_integ->AddDomainGrad(current_x, -1, *Base_::Grad);
    }

    // boundary
    for (auto& boundary_integ : boundary_face_nfi_) {
      boundary_integ->dt_ = dt_;
      boundary_integ->first_effective_dt_ = first_effective_dt_;
      boundary_integ->second_effective_dt_ = second_effective_dt_;
      boundary_integ->AddBoundaryGrad(current_x, -1, *Base_::Grad);
    }

    if (!Base_::Grad->Finalized()) {
      Base_::Grad->Finalize(0 /* skip_zeros */);
    }

    // set true dofs - if we have time, we could use nthread this (?)
    for (const auto& tdof : Base_::ess_tdof_list) {
      Base_::Grad->EliminateRowCol(tdof);
    }

    return *Base_::Grad;
  }

  void AddDomainIntegrator(const NFIPointer_& nlfi) {
    MIMI_FUNC()

    domain_nfi_.push_back(nlfi);
  }

  void AddBdrFaceIntegrator(const NFIPointer_& nlfi,
                            const mfem::Array<int>* bdr_marker = nullptr) {
    MIMI_FUNC()

    boundary_face_nfi_.push_back(nlfi);
    boundary_markers_.push_back(bdr_marker);
  }

  void SetEssentialTrueDofs(const mfem::Array<int>& ess_true_dof_list) {
    MIMI_FUNC()

    Base_::SetEssentialTrueDofs(ess_true_dof_list);
  }
};

} // namespace mimi::forms
