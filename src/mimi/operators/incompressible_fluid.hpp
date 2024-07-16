#pragma once

#include <memory>

#include <mfem.hpp>

#include "mimi/operators/base.hpp"
#include "mimi/utils/containers.hpp"

namespace mimi::operators {

class IncompressibleFluid : public OperatorTwoBases {
public:
  using MimiBase_ = OperatorTwoBases;
//   using MfemBase_ = mfem::FirstOrderTimeDependentOperator;
  using LinearFormPointer_ = MimiBase_::LinearFormPointer_;
  using BilinearFormPointer_ = MimiBase_::BilinearFormPointer_;
  using MixedBilinearFormPointer_ = MimiBase_::MixedBilinearFormPointer_;

protected:
  // linear forms
  LinearFormPointer_ rhs_;

  // bilinear forms
  // BilinearFormPointer_ mass_;
  BilinearFormPointer_ diffusion_;
  MixedBilinearFormPointer_ conservation_;

  // pure vertex-wise forces - usually from fsi
  std::shared_ptr<mfem::Vector> rhs_vector_;
  // nonlinear stiffness -> material/body behavior

  // internal values - to set params for each implicit term
  const mfem::Vector* vel_;
  const mfem::Vector* p_;
  double fac0_;
  double fac1_;

  // unlike base classes, we will keep one sparse matrix and initialize
  std::unique_ptr<mfem::SparseMatrix> owning_jacobian_;

  /// data to mass sparse matrix - A is mfem's notation of data array
  // const double* mass_A_ = nullptr;
  // int mass_n_nonzeros_ = -1;

  mutable mfem::Vector temp_vel_;
  mutable mfem::Vector temp_p_;

  mutable mfem::SparseMatrix* jacobian_ = nullptr;

public:
  /// This is same as Base_'s ctor
  IncompressibleFluid(
    mfem::FiniteElementSpace& fe_space_velocity,
    mfem::FiniteElementSpace& fe_space_pressure
    ) : MimiBase_(fe_space_velocity, fe_space_pressure) {
    MIMI_FUNC()
  }

  virtual std::string Name() const { return "IncompressibleFluid"; }

  virtual void SetParameters(double const& fac0,
                             double const& fac1,
                             const mfem::Vector* vel,
                             const mfem::Vector* p) {
    MIMI_FUNC()

    // this is from base
    fac0_ = fac0;
    fac1_ = fac1;
    vel_ = vel;
    p_ = p;

    // nonlinear_stiffness_->dt_ = dt_;
    // nonlinear_stiffness_->first_effective_dt_ = fac0;
    // nonlinear_stiffness_->second_effective_dt_ = fac1;
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

  // Bilinear form for the weak form of the pressure gradient and the mass conservation
  // TODO: check if it's correct
  virtual void SetupMixedBilinearConservationForm() {
    MIMI_FUNC()

    conservation_ = MimiBase_::mixed_bilinear_forms_["conservation"];
    if (conservation_) {
      mimi::utils::PrintAndThrowError("The conservation mixed bilinear form is not yet implemented!");
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
    SetupMixedBilinearConservationForm();
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

  // TODO: Mult for solving nonlinear system

  // TODO: GetGradient

  // TODO: ResidualAndGrad
};

} // namespace mimi::operators
