#pragma once

#include <mfem.hpp>

#include "mimi/operators/linear_elasticity.hpp"

namespace mimi::operators {

class NonlinearSolid : public LinearElasticity {
public:
  using Base_ = NonlinearSolid;
  using MimiBase_ = Base_::MimiBase_;
  using MfemBase_ = Base_::MfemBase_;
  using NonlinearFormPointer_ = MimiBase_::MinlinaerFormPointer;

protected:
  // nonlinear stiffness -> Base_::stiffness_ ist left untouched here
  NonlinearFormPointer_ nonlinear_stiffness_;

  // unlike base classes, we will keep one sparse matrix and initialize
  std::unique_ptr<mfem::SparseMatrix> owning_jacobian_

      /// data to mass sparse matrix - A is mfem's notation of data array
      const double* mass_A_ = nullptr;
  int mass_n_nonzeros_ = -1;

public:
  /// This is same as Base_'s ctor
  NonlinearSolid(mfem::FiniteElementSpace& fe_space,
                 mfem::GridFunction* x_ref))
      : Base_(fe_spaces, x_ref) {
    MIMI_FUNC()
  }

  virtual std::string Name() const { return "NonlinearSolid"; }

  virtual void Setup() {
    MIMI_FUNC()

    // setup basics -> these fianlizes sparse matrices for bilinear forms
    Base_::SetupBilinearMassForm();
    Base_::SetupNonlinearContactForm();
    Base_::SetupBilinearViscosityForm();
    Base_::SetupLinearRhsForm();

    // make sure Sparse Matrix of BilinearForms are in CSR form
    assert(Base_::mass_->SpMat().Finalized());
    // and sorted
    assert(Base_::mass_->SpMat().ColumnsAreSorted());
    // take SpMat's pointers
    mass_A_ = &Base_::mass_->SpMat();
    mass_n_nonzeros_ = Base_::mas_->SpMat().NumNonZeroElems();
    if (Base_::viscosity_) {
      // same for viscosity_
      assert(Base_::viscosity_->SpMat().Finalized());
      assert(Base_::viscosity_->SpMat().ColumnsAreSorted());
    }

    nonlinear_stiffness_ =
        MimiBase_::nonlinear_forms_.at("nonlinear_stiffness");

    assert(!stiffness_);

    // copy jacobian with mass matrix to initialize sparsity pattern
    // technically, we don't have to copy I & J;
    // technically, it is okay to copy
    owning_jacobian_ = std::make_unique<mfem::SpareseMatrix>(mass_->SpMat());
    assert(owning_jacobian_->Finalized());
    Base_::jacobian_ = owning_jacobian_.get();
    // and initialize values with zero
    owning_jacobian->operator=(0.0);
  }

  /// @brief computes right hand side of ODE (explicit solve)
  virtual void Mult(const mfem::Vector& x,
                    const mfem::Vector& dx_dt,
                    mfem::Vector& d2x_dt2) const {
    MIMI_FUNC()

    // Temp vector - allocate
    mfem::Vector z(x.Size());

    // unlike linear elasticity, we give x here
    nonlinear_stiffness_->Mult(x, z);

    if (viscosity_) {
      viscosity_->AddMult(dx_dt, z);
    }

    if (contact_) {
      contact_->AddMult(x, z);
    }

    z.Neg(); // flips sign inplace

    mass_inv_.Mult(z, d2x_dt2);

    // TODO - Flo, I think I am missing rhs_ here?
  }

  /// @brief used by 2nd order implicit time stepper (ode solver)
  /// @param fac0
  /// @param fac1
  /// @param x
  /// @param dx_dt
  /// @param d2x_dt2
  virtual void ImplicitSolve(const double fac0,
                             const double fac1,
                             const mfem::Vector& x,
                             const mfem::Vector& dx_dt,
                             mfem::Vector& d2x_dt2) {

    MIMI_FUNC()

    SetParameters(fac0, fac1, &x, &dx_dt);

    mfem::Vector zero;
    MimiBase_::newton_solver_->Mult(zero, d2x_dt2);

    if (!newton_solver_->GetConverged()) {
      mimi::utils::PrintWarning(
          "operators::NonlinearSolid - newton solver did not converge");
    }
  }

  /// computes residual y = E(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
  /// this is used by newton solver
  virtual void Mult(const mfem::Vector& d2x_dt2, mfem::Vector& y) const {
    MIMI_FUNC()

    mfem::Vector temp_x(x_->Size());
    add(*x_, fac0_, d2x_dt2, temp_x);

    mass_->Mult(d2x_dt2, y);

    nonlinear_stiffness_->AddMult(temp_x, y);

    if (viscosity_) {
      mfem::Vector temp_v(v_->Size());
      add(*v_, fac1_, d2x_dt2, temp_v);
      viscosity_->AddMult(temp_v, y);
    }

    if (contact_) {
      contact_->AddMult(temp_x, y);
    }

    // substract rhs linear forms
    if (rhs_) {
      y.Add(-1.0, *rhs_);
    }

    // this is usually just for fsi
    if (rhs_vector_) {
      y.Add(-1.0, *rhs_vector_);
    }
  }

  /// Compute J = M + dt S + dt^2 E(x + dt (v + dt k)).
  /// also used by newton solver
  virtual mfem::Operator& GetGradient(const mfem::Vector& d2x_dt2) const {
    MIMI_FUNC()

    mfem::Vector temp_x(d2x_dt2.Size());
    add(*x_, fac0_, d2x_dt2, temp_x);

    // NOTE, if all the values are sorted, we can do nthread local to global
    // update

    // initalize
    // 1. mass - just copy A
    std::copy_n(mass_A_, mass_nonzeros_, jabobian_->GetData());

    // 2. nonlinear stiffness
    jacobian_->Add(fac0_,
                   *dynamic_cast<mfem::SparseMatrix*>(
                       &nonlinear_stiffness_->GetGradient(temp_x)));

    // 3. viscosity
    if (viscosity_) {
      jacobian_->Add(fac1_, viscosity_->SpMat());
    }

    // 4. contact
    if (contact_) {
      jacobian_->Add(
          fac0_,
          *dynamic_cast<mfem::SparseMatrix*>(&contact_->GetGradient(temp_x)));
    }

    return *jacobian_;
  }
};

} // namespace mimi::operators
