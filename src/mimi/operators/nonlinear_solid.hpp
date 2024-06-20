#pragma once

#include <memory>

#include <mfem.hpp>

#include "mimi/operators/linear_elasticity.hpp"
#include "mimi/utils/containers.hpp"

namespace mimi::operators {

class NonlinearSolid : public LinearElasticity {
public:
  using Base_ = LinearElasticity;
  using MimiBase_ = Base_::MimiBase_;
  using MfemBase_ = Base_::MfemBase_;
  using NonlinearFormPointer_ = MimiBase_::NonlinearFormPointer_;

protected:
  // nonlinear stiffness -> Base_::stiffness_ ist left untouched here
  NonlinearFormPointer_ nonlinear_stiffness_;

  // unlike base classes, we will keep one sparse matrix and initialize
  std::unique_ptr<mfem::SparseMatrix> owning_jacobian_;

  /// data to mass sparse matrix - A is mfem's notation of data array
  const double* mass_A_ = nullptr;
  int mass_n_nonzeros_ = -1;

  mutable mfem::Vector temp_x_;
  mutable mfem::Vector temp_v_;

public:
  /// This is same as Base_'s ctor
  NonlinearSolid(mfem::FiniteElementSpace& fe_space, mfem::GridFunction* x_ref)
      : Base_(fe_space, x_ref) {
    MIMI_FUNC()
  }

  virtual std::string Name() const { return "NonlinearSolid"; }

  virtual void SetParameters(double const& fac0,
                             double const& fac1,
                             const mfem::Vector* x,
                             const mfem::Vector* v) {
    MIMI_FUNC()

    // this is from base
    fac0_ = fac0;
    fac1_ = fac1;
    x_ = x;
    v_ = v;

    nonlinear_stiffness_->dt_ = dt_;
    nonlinear_stiffness_->first_effective_dt_ = fac0;
    nonlinear_stiffness_->second_effective_dt_ = fac1;
  }

  virtual void SetupDirichletDofsFromNonlinearStiffness() {
    MIMI_FUNC()

    dirichlet_dofs_ = &nonlinear_stiffness_->GetEssentialTrueDofs();
  }

  virtual void Setup() {
    MIMI_FUNC()

    // setup basics -> these finalizes sparse matrices for bilinear forms
    Base_::SetupBilinearMassForm();
    Base_::SetupNonlinearContactForm();
    Base_::SetupBilinearViscosityForm();
    Base_::SetupLinearRhsForm();

    // make sure Sparse Matrix of BilinearForms are in CSR form
    assert(Base_::mass_->SpMat().Finalized());
    // and sorted
    assert(Base_::mass_->SpMat().ColumnsAreSorted());
    // take SpMat's pointers
    mass_A_ = Base_::mass_->SpMat().GetData();
    mass_n_nonzeros_ = Base_::mass_->SpMat().NumNonZeroElems();
    if (Base_::viscosity_) {
      // same for viscosity_
      assert(Base_::viscosity_->SpMat().Finalized());
      assert(Base_::viscosity_->SpMat().ColumnsAreSorted());
    }

    nonlinear_stiffness_ =
        MimiBase_::nonlinear_forms_.at("nonlinear_stiffness");

    SetupDirichletDofsFromNonlinearStiffness();

    assert(!stiffness_);

    // copy jacobian with mass matrix to initialize sparsity pattern
    // technically, we don't have to copy I & J;
    // technically, it is okay to copy
    owning_jacobian_ = std::make_unique<mfem::SparseMatrix>(mass_->SpMat());
    assert(owning_jacobian_->Finalized());
    Base_::jacobian_ = owning_jacobian_.get();
    // and initialize values with zero
    owning_jacobian_->operator=(0.0);
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
      contact_->AddMult(x, z); // we have flipped the sign at integrator
    }

    z.Neg(); // flips sign inplace

    // substract rhs linear forms
    if (rhs_) {
      z += *rhs_;
    }

    // this is usually just for fsi
    if (rhs_vector_) {
      z += *rhs_vector_;
    }

    mass_inv_.Mult(z, d2x_dt2);
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

    // if (!newton_solver_->GetConverged()) {
    //   mimi::utils::PrintWarning(
    //       "operators::NonlinearSolid - newton solver did not converge");
    // }
  }

  /// computes residual y = E(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
  /// this is used by newton solver
  virtual void Mult(const mfem::Vector& d2x_dt2, mfem::Vector& y) const {
    MIMI_FUNC()

    temp_x_.SetSize(x_->Size());
    add(*x_, fac0_, d2x_dt2, temp_x_);

    mass_->Mult(d2x_dt2, y);

    nonlinear_stiffness_->AddMult(temp_x_, y);

    if (viscosity_) {
      temp_v_.SetSize(v_->Size());
      add(*v_, fac1_, d2x_dt2, temp_v_);
      viscosity_->AddMult(temp_v_, y);
    }

    if (contact_) {
      contact_->AddMult(temp_x_, y);
    }

    // substract rhs linear forms
    if (rhs_) {
      y.Add(-1.0, *rhs_);
    }

    // this is usually just for fsi
    if (rhs_vector_) {
      y.Add(-1.0, *rhs_vector_);
    }

    for (const int i : *dirichlet_dofs_) {
      y[i] = 0.0;
    }
  }

  /// Compute J = M + dt S + dt^2 E(x + dt (v + dt k)).
  /// also used by newton solver
  virtual mfem::Operator& GetGradient(const mfem::Vector& d2x_dt2) const {
    MIMI_FUNC()

    temp_x_.SetSize(x_->Size());
    add(*x_, fac0_, d2x_dt2, temp_x_);

    // NOTE, if all the values are sorted, we can do nthread local to global
    // update
    double* J_data = jacobian_->GetData();
    const double* K_data = dynamic_cast<mfem::SparseMatrix*>(
                               &nonlinear_stiffness_->GetGradient(temp_x_))
                               ->GetData();

    // get mass and nonlin stiff at the same time
    for (int i{}; i < mass_n_nonzeros_; ++i) {
      J_data[i] = mass_A_[i] + (fac0_ * K_data[i]);
    }

    // 3. viscosity
    if (viscosity_) {
      jacobian_->Add(fac1_, viscosity_->SpMat());
    }

    // 4. contact
    if (contact_) {
      jacobian_->Add(
          fac0_,
          *dynamic_cast<mfem::SparseMatrix*>(&contact_->GetGradient(temp_x_)));
    }

    return *jacobian_;
  }

  virtual mfem::Operator* ResidualAndGrad(const mfem::Vector& d2x_dt2,
                                          const int nthread,
                                          mfem::Vector& y) const {

    temp_x_.SetSize(x_->Size());
    add(*x_, fac0_, d2x_dt2, temp_x_);

    // do usual residual operation
    mass_->Mult(d2x_dt2, y); // this initializes y to zero
    if (viscosity_) {
      temp_v_.SetSize(v_->Size());
      add(*v_, fac1_, d2x_dt2, temp_v_);
      viscosity_->AddMult(temp_v_, y);
    }

    // now nonlin part
    // 1. initalize grad with mass
    std::copy_n(mass_A_, mass_n_nonzeros_, jacobian_->GetData());
    nonlinear_stiffness_->AddMultGrad(temp_x_, nthread, fac0_, y, *jacobian_);

    // contact -> both grad and residual
    if (contact_) {
      contact_->AddMultGrad(temp_x_, nthread, fac0_, y, *jacobian_);
    }

    // 3. viscosity
    if (viscosity_) {
      jacobian_->Add(fac1_, viscosity_->SpMat());
    }

    if (rhs_) {
      y.Add(-1.0, *rhs_);
    }
    // this is usually just for fsi
    if (rhs_vector_) {
      y.Add(-1.0, *rhs_vector_);
    }

    for (const int i : *dirichlet_dofs_) {
      y[i] = 0.0;
    }

    return jacobian_;
  }

  virtual void PostTimeAdvance(const mfem::Vector& x, const mfem::Vector& v) {
    MIMI_FUNC()

    nonlinear_stiffness_->PostTimeAdvance(x);
    if (contact_)
      contact_->PostTimeAdvance(x);
  }
};

} // namespace mimi::operators
