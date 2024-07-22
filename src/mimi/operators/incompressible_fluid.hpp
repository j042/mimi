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

  // bilinear forms
  NonlinearFormPointer_ stokes;

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

  // TODO: Surely wrong; check if this is right
  virtual void SetupDirichletDofsFromBilinearForms() {
    MIMI_FUNC()

    dirichlet_dofs_velocity_ = &diffusion_->GetEssentialTrueDofs();
    dirichlet_dofs_pressure_ = &conservation_->GetEssentialTrueDofs();
  }

  virtual void SetRhsVector(const std::shared_ptr<mfem::Vector>& rhs_vector) {
    MIMI_FUNC()

    rhs_vector_ = rhs_vector;
  }

  // TODO: Check
  virtual void SetupBilinearDiffusionForm() {
    MIMI_FUNC()

    // Diffusion
    diffusion_ = MimiBase_::bilinear_forms_["diffusion"];
    if (diffusion_) {
      diffusion_->Finalize(0); // skip_zero is 0
      // if this is sorted, we can just add A
      diffusion_->SpMat().SortColumnIndices();
      mimi::utils::PrintInfo(Name(), "has diffusion term.");
    }
  }

  // Bilinear form for the weak form of the pressure gradient and the mass
  // conservation
  // TODO: check if it's correct
  virtual void SetupBilinearConservationForm() {
    MIMI_FUNC()

    conservation_ = MimiBase_::bilinear_forms_["conservation"];
    if (conservation_) {
      mimi::utils::PrintAndThrowError(
          "The conservation bilinear form is not yet implemented!");
    }
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

    // setup basics -> these finalizes sparse matrices for bilinear forms
    SetupBilinearDiffusionForm();
    SetupBilinearConservationForm();
    SetupLinearRhsForm();

    if (diffusion_) {
      // same for diffusion_
      assert(diffusion_->SpMat().Finalized());
      assert(diffusion_->SpMat().ColumnsAreSorted());
    }
    // TODO: same for conservation

    SetupDirichletDofsFromBilinearForms();

    // TODO: what does the following do?
    // assert(!stiffness_);

    // copy jacobian with mass matrix to initialize sparsity pattern
    // technically, we don't have to copy I & J;
    // technically, it is okay to copy
    owning_jacobian_ = std::make_unique<mfem::SparseMatrix>(mass_->SpMat());
    assert(owning_jacobian_->Finalized());
    jacobian_ = owning_jacobian_.get();
    // and initialize values with zero
    owning_jacobian_->operator=(0.0);
  }

  // TODO: time dependent ODE solve

  // TODO
  /// @brief Implicit solver
  virtual void ImplicitSolve(mfem::Vector& vel, mfem::Vector& p) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError(
        "Still unsure what implicit solve should output!");
  }

  // Mult for solving nonlinear system
  // computes residual, which is used by Newton solver
  // TODO: check Dirichlet Dofs
  virtual void Mult(mfem::Vector& residual) const {
    MIMI_FUNC()
    temp_res_vel_.SetSize(vel_->Size());
    temp_res_p_.SetSize(p_->Size());

    // A v
    diffusion_->AddMult(vel_, temp_res_vel_);

    // should be right; otherwise just swap
    // div(v) q
    conservation_->AddMult(vel_, temp_res_vel_);
    // div(w) p
    conservation_->AddMultTranspose(p_, temp_res_p_);

    // substract rhs
    if (rhs_) {
      residual.Add(-1.0, *rhs_);
    }

    // this is usually just for fsi
    if (rhs_vector_) {
      residual.Add(-1.0, *rhs_vector_);
    }

    // Set residual of Dirichlet DoFs to zero
    for (const int i : *all_dirichlet_dofs) {
      residual[i] = 0.0;
    }
    // ------------
    // TODO: declare ndofs: should be dofs vel + dofs p
    // copy or move??? to residual vector???
  }

  // TODO: GetGradient
  virtual mfem::Operator& GetGradient() const {
    MIMI_FUNC()

    // TODO nothing more?

    return *jacobian_;
  }

  // TODO: ResidualAndGrad
  virtual mfem::Operator* ResidualAndGrad(const int nthread,
                                          mfem::Vector& residual) const {
    // Residual
    // A v
    diffusion_->AddMult(vel_, residual);

    // should be right; otherwise just swap
    // div(v) q
    conservation_->AddMult(vel_, residual);
    // div(w) p
    conservation_->AddMultTranspose(p_, residual);

    // substract rhs
    if (rhs_) {
      residual.Add(-1.0, *rhs_);
    }

    // this is usually just for fsi
    if (rhs_vector_) {
      residual.Add(-1.0, *rhs_vector_);
    }

    // Set residual of Dirichlet DoFs to zero
    for (const int i : *dirichlet_dofs_velocity_) {
      residual[i] = 0.0;
    }
    for (const int i : *dirichlet_dofs_pressure_) {
      residual[i] = 0.0;
    }

    return jacobian_;
  }
};

} // namespace mimi::operators
