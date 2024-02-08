#pragma once

#include <mfem.hpp>

#include "mimi/forms/nonlinear_visco.hpp"
#include "mimi/operators/nonlinear_solid.hpp"

namespace mimi::operators {

class NonlinearViscoSolid : public NonlinearSolid {
public:
  using Base_ = NonlinearSolid;
  using MimiBase_ = Base_::MimiBase_;
  using MfemBase_ = Base_::MfemBase_;
  using NonlinearFormPointer_ = std::shared_ptr<mimi::forms::NonlinearVisco>;

protected:
  // nonlinear visco stiffness -> Base_::nonlinear_stiffness_ is left untouched
  // here
  NonlinearFormPointer_ nonlinear_visco_stiffness_;

  // unlike base classes, we will keep one sparse matrix and initialize
  std::unique_ptr<mfem::SparseMatrix> owning_jacobian_;

  /// data to mass sparse matrix - A is mfem's notation of data array
  const double* mass_A_ = nullptr;
  int mass_n_nonzeros_ = -1;

  mutable mfem::Vector temp_x;
  mutable mfem::Vector temp_v;

public:
  using Base_::Base_;

  virtual std::string Name() const { return "NonlinearViscoSolid"; }

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
    mass_A_ = Base_::mass_->SpMat().GetData();
    mass_n_nonzeros_ = Base_::mass_->SpMat().NumNonZeroElems();
    if (Base_::viscosity_) {
      // same for viscosity_
      assert(Base_::viscosity_->SpMat().Finalized());
      assert(Base_::viscosity_->SpMat().ColumnsAreSorted());
    }

    nonlinear_visco_stiffness_ =
        std::dynamic_pointer_cast<NonlinearFormPointer_::element_type>(
            MimiBase_::nonlinear_forms_.at("nonlinear_visco_stiffness"));

    assert(!stiffness_);
    assert(!nonlinear_stiffness_);
    assert(nonlinear_visco_stiffness_);

    // we upcast to nonlinear_stiffness to avoid re-implementing
    // set parameters and freeze/melt
    nonlinear_stiffness_ = nonlinear_visco_stiffness_;
    SetupDirichletDofsFromNonlinearStiffness();

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

    // unlike nonlinear solids, we give x and x_dot here
    nonlinear_visco_stiffness_->Mult(x, dx_dt, z);

    if (viscosity_) {
      viscosity_->AddMult(dx_dt, z);
    }

    if (contact_) {
      contact_->AddMult(x, z);
    }

    // substract rhs linear forms
    if (rhs_) {
      z += *rhs_;
    }

    // this is usually just for fsi
    if (rhs_vector_) {
      z += *rhs_vector_;
    }

    z.Neg(); // flips sign inplace

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

    if (!newton_solver_->GetConverged()) {
      mimi::utils::PrintWarning(
          "operators::NonlinearSolid - newton solver did not converge");
    }
  }

  /// computes residual y = E(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
  /// this is used by newton solver
  virtual void Mult(const mfem::Vector& d2x_dt2, mfem::Vector& y) const {
    MIMI_FUNC()

    // std::cout << "\n\n\n\nd2xdt2[0] " << d2x_dt2[0] << std::endl;
    // std::cout << "d2xdt2[1] " << d2x_dt2[2] << std::endl;
    // std::cout << "d2xdt2[2] " << d2x_dt2[4] << std::endl;
    // std::cout << "d2xdt2[3] " << d2x_dt2[6] << std::endl;

    temp_x.SetSize(x_->Size());
    add(*x_, fac0_, d2x_dt2, temp_x);

    // for (const int& d_id : *dirichlet_dofs_) {
    //   temp_x[d_id] = (*x_)[d_id];
    // }

    temp_v.SetSize(v_->Size());
    add(*v_, fac1_, d2x_dt2, temp_v);

    // std::cout << "temp_x[0] " << temp_x[0] << std::endl;
    // std::cout << "temp_x[1] " << temp_x[2] << std::endl;
    // std::cout << "temp_x[2] " << temp_x[4] << std::endl;
    // std::cout << "temp_x[3] " << temp_x[6] << std::endl;
    // std::cout << "x[0] " << (*x_)[0] << std::endl;
    // std::cout << "x[1] " << (*x_)[2] << std::endl;
    // std::cout << "x[2] " << (*x_)[4] << std::endl;
    // std::cout << "x[3] " << (*x_)[6] << std::endl;

    mass_->Mult(d2x_dt2, y);

    nonlinear_visco_stiffness_->AddMult(temp_x, temp_v, y);
    // std::cout << "y[0] " << y[0] << std::endl;
    // std::cout << "y[1] " << y[2] << std::endl;
    // std::cout << "y[2] " << y[4] << std::endl;
    // std::cout << "y[3] " << y[6] << std::endl;

    if (viscosity_) {
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

    temp_x.SetSize(d2x_dt2.Size());
    add(*x_, fac0_, d2x_dt2, temp_x);

    // if we just assemble grad during residual, we can remove this
    temp_v.SetSize(v_->Size());
    add(*v_, fac1_, d2x_dt2, temp_v);

    // NOTE, if all the values are sorted, we can do nthread local to global
    // update

    // initalize
    // 1. mass - just copy A
    std::copy_n(mass_A_, mass_n_nonzeros_, jacobian_->GetData());

    // 2. nonlinear stiffness
    jacobian_->Add(
        fac0_,
        *dynamic_cast<mfem::SparseMatrix*>(
            &nonlinear_visco_stiffness_->GetGradient(temp_x, temp_v)));

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
