#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <mfem.hpp>

#include "mimi/operators/base.hpp"

namespace mimi::operators {

class LinearElasticity : public OperatorBase,
                         public mfem::SecondOrderTimeDependentOperator {
public:
  using MimiBase_ = OperatorBase;
  using MfemBase_ = mfem::SecondOrderTimeDependentOperator;

protected:
  // linear forms
  typename MimiBase_::LinearFormPointer_ rhs_;

  // bilinear forms
  typename MimiBase_::BilinearFormPointer_ mass_;
  typename MimiBase_::BilinearFormPointer_ viscosity_;
  typename MimiBase_::BilinearFormPointer_ stiffness_;

  // non linear forms
  typename MimiBase_::NonlinearFormPointer_ contact_;

  // pure vertex-wise forces - usually from fsi
  std::shared_ptr<mfem::Vector> rhs_vector_;

  // mass matrix inversion using krylov solver
  mfem::CGSolver mass_inv_;
  mfem::DSmoother mass_inv_prec_;
  double rel_tol_{1e-8};
  double abs_tol_{1e-12};
  int max_iter_{1000};

  // internal values - to set params for each implicit term
  const mfem::Vector* x_;
  const mfem::Vector* v_;
  double fac0_;
  double fac1_;

  // holder for jacobian. Add() creates a new jacobian and we need to be able to
  // destroy it afterwards
  mfem::SparseMatrix* jacobian_;

public:
  /// @brief ctor
  ///
  /// dv/dt = -M^{-1}*(E(x) + S*v)
  /// dx/dt = v,
  /// S is viscosity/damping matrix, E is elasticity stiffness matrix, and M is
  /// mass matrix
  /// @param fe_space
  LinearElasticity(mfem::FiniteElementSpace& fe_space)
      : MimiBase_(fe_space),
        MfemBase_(fe_space.GetTrueVSize(), 0.0) {
    MIMI_FUNC();
  }

  virtual std::string Name() { return "LinearElasticity"; }

  virtual void
  SolverConfig(const double rel_tol, const double abs_tol, const int max_iter) {
    MIMI_FUNC()

    rel_tol_ = rel_tol;
    abs_tol_ = abs_tol;
    max_iter_ = max_iter;
  }

  virtual void SetRhsVector(const std::shared_ptr<mfem::Vector>& rhs_vector) {
    MIMI_FUNC()

    rhs_vector_ = rhs_vector;
  }

  /// @brief setup all the necessary forms
  virtual void Setup() {
    MIMI_FUNC()

    // setup mass matrix and inverter
    mass_ = MimiBase_::bilinear_forms_.at("mass");
    mass_inv_.iterative_mode = false;
    mass_inv_.SetRelTol(rel_tol_);
    mass_inv_.SetAbsTol(abs_tol_);
    mass_inv_.SetMaxIter(max_iter_);
    mass_inv_.SetPrintLevel(
        mfem::IterativeSolver::PrintLevel().Warnings().Errors().Summary());
    mass_inv_.SetPreconditioner(mass_inv_prec_);
    mass_inv_.SetOperator(mass_->SpMat());

    // stiffness
    stiffness_ = MimiBase_::bilinear_forms_.at("stiffness");

    // contact
    contact_ = MimiBase_::nonlinear_forms_["contact"];
    if (contact_) {
      mimi::utils::PrintInfo(Name(), "has contact term.");
    }

    // following forms are optional
    // viscosity
    viscosity_ = MimiBase_::bilinear_forms_["viscosity"];
    if (viscosity_) {
      mimi::utils::PrintInfo(Name(), "has viscosity term.");
    }

    // rhs linear form
    rhs_ = MimiBase_::linear_forms_["rhs"];
    if (rhs_) {
      mimi::utils::PrintInfo(Name(), "has rhs linear form term");
    }
  };

  virtual void SetParameters(double const& fac0,
                             double const& fac1,
                             const mfem::Vector* x,
                             const mfem::Vector* v) {
    MIMI_FUNC()
    fac0_ = fac0;
    fac1_ = fac1;
    x_ = x;
    v_ = v;
  }

  /// @brief computes right hand side of ODE (explicit solve)
  virtual void Mult(const mfem::Vector& x,
                    const mfem::Vector& dx_dt,
                    mfem::Vector& d2x_dt2) {
    MIMI_FUNC()

    // Temp vector
    mfem::Vector z(x.Size());

    stiffness_->Mult(x, z);

    if (viscosity_) {
      viscosity_->AddMult(dx_dt, z);
    }

    if (contact_) {
      contact_->AddMult(x, z);
    }

    z.Neg(); // flips sign inplace
    mass_inv_.Mult(z, d2x_dt2);
  }

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
          "operators::LinearElasticity - newton solver did not converge");
    }
  }

  /// computes residual y = E(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
  virtual void Mult(const mfem::Vector& d2x_dt2, mfem::Vector& y) const {
    MIMI_FUNC()
    mfem::Vector temp_x(x_->Size());
    add(*x_, fac0_, d2x_dt2, temp_x);

    mass_->Mult(d2x_dt2, y);

    stiffness_->AddMult(temp_x, y);

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
  mfem::Operator& GetGradient(const mfem::Vector& d2x_dt2) {
    MIMI_FUNC()

    mfem::Vector temp_x(d2x_dt2.Size());
    add(*x_, fac0_, d2x_dt2, temp_x);

    // release;
    if (jacobian_)
      delete jacobian_;

    jacobian_ = mfem::Add(1.0, mass_->SpMat(), fac0_, stiffness_->SpMat());

    if (viscosity_) {
      jacobian_->Add(fac1_, viscosity_->SpMat());
    }

    if (contact_) {
      jacobian_->Add(
          fac0_,
          *dynamic_cast<mfem::SparseMatrix*>(&contact_->GetGradient(temp_x)));
    }

    return *jacobian_;
  }
};

} // namespace mimi::operators
