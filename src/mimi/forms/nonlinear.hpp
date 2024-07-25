#pragma once

#include <memory>

#include <mfem.hpp>

#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::forms {
/* Extends Mult() to incorporate mimi::integrators::nonloinear_base */
class Nonlinear : public mfem::Operator {
protected:
  const mfem::Array<int>* dirichlet_dofs_;

  // mutable to satisfy const-ness of GetGradient()
  mutable std::unique_ptr<mfem::SparseMatrix> grad_;

  mfem::SparseMatrix& InitializeGrad() const {
    if (!grad_) {
      // this is an adhoc solution to get sparsity pattern
      // we know one of nl integrator should have precomputed, so access matrix
      // from that
      mfem::SparseMatrix* sparsity_pattern{};
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

      grad_ = std::make_unique<mfem::SparseMatrix>(*sparsity_pattern);
      *grad_ = 0.0;
    } else {
      *grad_ = 0.0;
    }

    return *grad_;
  }

public:
  using Base_ = mfem::Operator;
  using NFIPointer_ = std::shared_ptr<mimi::integrators::NonlinearBase>;

  mimi::utils::Vector<NFIPointer_> domain_nfi_{};
  mimi::utils::Vector<NFIPointer_> boundary_face_nfi_{};
  mimi::utils::Vector<const mfem::Array<int>*> boundary_markers_{};

  /// for single fespace, we inherit ctor
  using Base_::Base_;

  /// used for custom spaces - e.g., for stokes
  Nonlinear(const int true_v_size) : Base_(true_v_size) {}

  /// time step size, in case you need them
  /// operators should set them
  double dt_{0.0};
  double first_effective_dt_{0.0};  // this is for x
  double second_effective_dt_{0.0}; // this is for x_dot (=v).

  virtual void PostTimeAdvance(const mfem::Vector& x) {
    MIMI_FUNC()

    // currently, we don't do boundaries
    for (auto& dnfi : domain_nfi_) {
      dnfi->DomainPostTimeAdvance(x);
    }
    for (auto& bfnfi : boundary_face_nfi_) {
      bfnfi->BoundaryPostTimeAdvance(x);
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
                       mfem::Vector& residual,
                       const double factor = 1.0) const {
    MIMI_FUNC()
    // currently ignored
    assert(factor == 1.0);

    // we assemble all first - these will call nthreadexe
    // domain
    for (auto& domain_integ : domain_nfi_) {
      domain_integ->dt_ = dt_;
      domain_integ->first_effective_dt_ = first_effective_dt_;
      domain_integ->second_effective_dt_ = second_effective_dt_;
      domain_integ->AddDomainResidual(current_x, residual);
    }

    // boundary
    for (auto& boundary_integ : boundary_face_nfi_) {
      boundary_integ->dt_ = dt_;
      boundary_integ->first_effective_dt_ = first_effective_dt_;
      boundary_integ->second_effective_dt_ = second_effective_dt_;
      boundary_integ->AddBoundaryResidual(current_x, residual);
    }

    // set true dofs - if we have time, we could use nthread this.
    if (dirichlet_dofs_) {
      for (const auto& tdof : GetDirichletDofs()) {
        residual[tdof] = 0.0;
      }
    }
  }

  virtual void AddMultGrad(const mfem::Vector& current_x,
                           const int nthread,
                           const double grad_factor,
                           mfem::Vector& residual,
                           mfem::SparseMatrix& grad) const {
    MIMI_FUNC()

    for (auto& domain_integ : domain_nfi_) {
      domain_integ->dt_ = dt_;
      domain_integ->first_effective_dt_ = first_effective_dt_;
      domain_integ->second_effective_dt_ = second_effective_dt_;
      domain_integ->AddDomainResidualAndGrad(current_x,
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
                                                 grad_factor,
                                                 residual,
                                                 grad);
    }

    // set true dofs - if we have time, we could use nthread this.
    if (dirichlet_dofs_) {
      for (const auto& tdof : GetDirichletDofs()) {
        residual[tdof] = 0.0;
        grad.EliminateRowCol(tdof);
      }
    }
  }

  virtual mfem::Operator& GetGradient(const mfem::Vector& current_x) const {
    MIMI_FUNC();

    InitializeGrad();

    // we assemble all first - these will call nthreadexe
    // domain
    for (auto& domain_integ : domain_nfi_) {
      domain_integ->dt_ = dt_;
      domain_integ->first_effective_dt_ = first_effective_dt_;
      domain_integ->second_effective_dt_ = second_effective_dt_;
      domain_integ->AddDomainGrad(current_x, *grad_);
    }

    // boundary
    for (auto& boundary_integ : boundary_face_nfi_) {
      boundary_integ->dt_ = dt_;
      boundary_integ->first_effective_dt_ = first_effective_dt_;
      boundary_integ->second_effective_dt_ = second_effective_dt_;
      boundary_integ->AddBoundaryGrad(current_x, *grad_);
    }

    if (!grad_->Finalized()) {
      grad_->Finalize(0 /* skip_zeros */);
    }

    // set true dofs - if we have time, we could use nthread this (?)
    if (dirichlet_dofs_) {
      for (const auto& tdof : GetDirichletDofs()) {
        grad_->EliminateRowCol(tdof);
      }
    }

    return *grad_;
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

  void SetDirichletDofs(const mfem::Array<int>& d_dofs) {
    MIMI_FUNC()
    dirichlet_dofs_ = &d_dofs;
  }

  const mfem::Array<int>& GetDirichletDofs() const {
    MIMI_FUNC()
    if (!dirichlet_dofs_) {
      mimi::utils::PrintAndThrowError(
          "mimi::forms::Nonlinear - Dirichlet dof does not exist.");
    }
    return *dirichlet_dofs_;
  }
};

} // namespace mimi::forms
