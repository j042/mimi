#pragma once

#include <memory>

#include <mfem.hpp>

#include "mimi/operators/base.hpp"
#include "mimi/utils/containers.hpp"

namespace mimi::operators {

class IncompressibleFluid : public OperatorBase,
                            public mfem::TimeDependentOperator {
public:
  using MfemBase_ = mfem::TimeDependentOperator;
  using MimiBase_ = OperatorBase;
  using LinearFormPointer_ = MimiBase_::LinearFormPointer_;
  using BilinearFormPointer_ = MimiBase_::BilinearFormPointer_;
  using NonlinearFormPointer_ = MimiBase_::NonlinearFormPointer_;

protected:
  // linear forms
  LinearFormPointer_ rhs_;

  // nonlinear form - it may be bilinear form in reality
  NonlinearFormPointer_ stokes_;

  // internal values - to set params for each implicit term
  const mfem::Vector* v_p_;

  // unlike base classes, we will keep one sparse matrix and initialize
  std::unique_ptr<mfem::SparseMatrix> owning_jacobian_;
  mutable mfem::SparseMatrix* jacobian_ = nullptr;

public:
  /// This is same as Base_'s ctor
  IncompressibleFluid(const int size) : MfemBase_(size), MimiBase_() {
    MIMI_FUNC()
  }

  virtual std::string Name() const { return "IncompressibleFluid"; }

  virtual void SetParameters(const mfem::Vector* v_p) {
    MIMI_FUNC()

    v_p_ = v_p;
  }

  virtual void SetupStokesForm() {
    MIMI_FUNC()

    stokes_ = MimiBase_::nonlinear_forms_["stokes"];
  }

  virtual void SetupLinearRhsForm() {
    MIMI_FUNC()

    // rhs linear form
    rhs_ = MimiBase_::linear_forms_["rhs"];
    if (rhs_) {
      mimi::utils::PrintInfo(Name(), "has rhs linear form term");
    }
  }

  virtual void Setup() {
    MIMI_FUNC()

    SetupStokesForm();
    SetupLinearRhsForm();

    // make sure there's stokes
    assert(!stokes_);

    SetSparsity(*stokes_->domain_nfi_[0]->precomputed_->sparsity_pattern_);

    dirichlet_dofs_ = &stokes_->GetDirichletDofs();
  }

  // TODO: time dependent ODE solve

  // TODO
  /// @brief Implicit solver
  virtual void ImplicitSolve(const mfem::Vector& v_p, mfem::Vector& p) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError(
        "Still unsure what implicit solve should output!");
  }

  // Mult for solving nonlinear system
  // computes residual, which is used by Newton solver
  virtual void Mult(const mfem::Vector& k, mfem::Vector& y) const {
    MIMI_FUNC()

    // currently this one is only implemented for static case, so no
    stokes_->Mult(k, y);

    if (rhs_) {
      y.Add(-1.0, *rhs_);
    }
  }

  // TODO: GetGradient
  virtual mfem::Operator& GetGradient() const {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("GetGradient not yet implemented!");

    return *jacobian_;
  }

  // TODO: ResidualAndGrad
  virtual mfem::Operator* ResidualAndGrad(const mfem::Vector& k,
                                          const int nthread,
                                          mfem::Vector& residual) const {
    // initialize
    owning_jacobian_->operator=(0.0);
    residual = 0.0;

    // mult grad
    stokes_->AddMultGrad(k, nthread, 1.0, residual, *jacobian_);

    // substract rhs
    if (rhs_) {
      residual.Add(-1.0, *rhs_);
    }

    return jacobian_;
  }

  virtual void PostTimeAdvance(const mfem::Vector& x, const mfem::Vector& v) {
    MIMI_FUNC()
    mimi::utils::PrintDebug(
        "IncompressibleFluid::PostTimeAdvance - Doing nothing");
  }
};

} // namespace mimi::operators
