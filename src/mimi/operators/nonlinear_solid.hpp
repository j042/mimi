#pragma once

#include <memory>

#include <mfem.hpp>

#include "mimi/operators/base.hpp"

namespace mimi::operators {

class NonlinearSolid : public OperatorBase,
                       public mfem::SecondOrderTimeDependentOperator {
public:
  using MimiBase_ = OperatorBase;
  using MfemBase_ = mfem::SecondOrderTimeDependentOperator;
  using LinearFormPointer_ = MimiBase_::LinearFormPointer_;
  using BilinearFormPointer_ = MimiBase_::BilinearFormPointer_;
  using NonlinearFormPointer_ = MimiBase_::NonlinearFormPointer_;

protected:
  // linear forms
  LinearFormPointer_ rhs_;

  // bilinear forms
  BilinearFormPointer_ mass_;
  BilinearFormPointer_ viscosity_;
  BilinearFormPointer_ stiffness_;

  // non linear forms
  NonlinearFormPointer_ contact_;

  // pure vertex-wise forces - usually from fsi
  std::shared_ptr<mfem::Vector> rhs_vector_;
  // nonlinear stiffness -> material/body behavior
  NonlinearFormPointer_ nonlinear_stiffness_;

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

  // unlike base classes, we will keep one sparse matrix and initialize
  std::unique_ptr<mfem::SparseMatrix> owning_jacobian_;

  /// data to mass sparse matrix - A is mfem's notation of data array
  const double* mass_A_ = nullptr;
  int mass_n_nonzeros_ = -1;

  mutable mfem::Vector temp_x_;
  mutable mfem::Vector temp_v_;

  mutable mfem::SparseMatrix* jacobian_ = nullptr;

public:
  /// This is same as Base_'s ctor
  NonlinearSolid(mfem::FiniteElementSpace& fe_space)
      : MimiBase_(fe_space),
        MfemBase_(fe_space.GetTrueVSize(), 0.0) {
    MIMI_FUNC()
  }

  virtual std::string Name() const { return "NonlinearSolid"; }

  virtual void SetParameters(double const& fac0,
                             double const& fac1,
                             const mfem::Vector* x,
                             const mfem::Vector* v);

  virtual void SetupDirichletDofsFromNonlinearStiffness() {
    MIMI_FUNC()

    dirichlet_dofs_ = &nonlinear_stiffness_->GetEssentialTrueDofs();
  }

  virtual void SetRhsVector(const std::shared_ptr<mfem::Vector>& rhs_vector) {
    MIMI_FUNC()

    rhs_vector_ = rhs_vector;
  }

  virtual void SetupBilinearMassForm();

  virtual void SetupNonlinearContactForm();

  virtual void SetupBilinearViscosityForm();

  virtual void SetupLinearRhsForm();

  virtual void Setup();

  /// @brief computes right hand side of ODE (explicit solve)
  virtual void Mult(const mfem::Vector& x,
                    const mfem::Vector& dx_dt,
                    mfem::Vector& d2x_dt2) const;

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
                             mfem::Vector& d2x_dt2);

  /// computes residual y = E(x + dt*(v + dt*k)) + M*k + S*(v + dt*k)
  /// this is used by newton solver
  virtual void Mult(const mfem::Vector& d2x_dt2, mfem::Vector& y) const;

  /// Compute J = M + dt S + dt^2 E(x + dt (v + dt k)).
  /// also used by newton solver
  virtual mfem::Operator& GetGradient(const mfem::Vector& d2x_dt2) const;

  virtual mfem::Operator* ResidualAndGrad(const mfem::Vector& d2x_dt2,
                                          const int nthread,
                                          mfem::Vector& y) const;

  virtual void PostTimeAdvance(const mfem::Vector& x, const mfem::Vector& v);
};

} // namespace mimi::operators
