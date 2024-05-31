#pragma once

#include <string>

#include <mfem.hpp>

#include "mimi/operators/base.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::solvers {

class OdeBase {
protected:
  // we keep casted mimi operator base to set dt_
  mimi::operators::OperatorBase* mimi_operator_;
  const mfem::Array<int>* dirichlet_dofs_{nullptr};

public:
  virtual ~OdeBase() = default;
  virtual std::string Name() const = 0;
  virtual void SetupDirichletDofs(const mfem::Array<int>* dirichlet_dofs) {
    MIMI_FUNC()

    dirichlet_dofs_ = dirichlet_dofs;
  }
  virtual void PrintInfo() {
    mimi::utils::PrintInfo("No detailed info for", Name());
  }

  virtual void SaveMimiDownCast2(mfem::SecondOrderTimeDependentOperator* oper) {
    mimi_operator_ = dynamic_cast<mimi::operators::OperatorBase*>(oper);
    if (!mimi_operator_) {
      mimi::utils::PrintAndThrowError(
          "Failed to cast mfem::SecondOrderTimeDependentOperator to "
          "mimi::operators::OperatorBase.");
    }
  }

  virtual void
  StepTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("StepTime2 is not implemented for", Name());
  }

  virtual void FixedPointSolve2(const mfem::Vector& x,
                                const mfem::Vector& dxdt,
                                double& t,
                                double& dt) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("FixedPointSolve2 is not implemented for",
                                    Name());
  }

  /// @brief x and dxdt should be current solution and advances are performed
  /// inplace.
  /// @param x
  /// @param dxdt
  /// @param t
  /// @param dt
  virtual void FixedPointAdvance2(mfem::Vector& x,
                                  mfem::Vector& dxdt,
                                  double& t,
                                  double& dt) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("FixedPointAdvance2 is not implemented for",
                                    Name());
  }

  virtual void
  AdvanceTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("AdvanceTime2 is not implemented for",
                                    Name());
  }
};

class GeneralizedAlpha2 : public mfem::GeneralizedAlpha2Solver, public OdeBase {
protected:
  double fac0_, fac1_, fac2_, fac3_, fac4_, fac5_, fac1_inv_, fac5_inv_;
  bool fixed_point_predict_alpha_level_{true};

public:
  using Base_ = mfem::GeneralizedAlpha2Solver;

  GeneralizedAlpha2() = default;

  GeneralizedAlpha2(mfem::SecondOrderTimeDependentOperator& oper,
                    double rho_inf = 0.25)
      : Base_(rho_inf),
        fixed_point_predict_alpha_level_{true} {

    Base_::Init(oper);
    ComputeFactors();
    SaveMimiDownCast2(&oper);
  }

  virtual void ComputeFactors() {
    fac0_ = (0.5 - (beta / alpha_m));
    fac1_ = alpha_f;
    fac1_inv_ = 1. / fac1_;
    fac2_ = alpha_f * (1.0 - (gamma / alpha_m));
    fac3_ = beta * alpha_f / alpha_m;
    fac4_ = gamma * alpha_f / alpha_m;
    fac5_ = alpha_m;
    fac5_inv_ = 1. / fac5_;
  }

  virtual std::string Name() const { return "GeneralizedAlpha2"; }

  virtual void PrintInfo() {
    MIMI_FUNC()

    mimi::utils::PrintInfo("Info for", Name());
    Base_::PrintProperties();
  }

  virtual void
  StepTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
    MIMI_FUNC()

    mimi_operator_->dt_ = dt;
    Base_::Step(x, dxdt, t, dt);
  }
  virtual void FixedPointSolve2(const mfem::Vector& x,
                                const mfem::Vector& dxdt,
                                double& t,
                                double& dt) {
    MIMI_FUNC()

    // In the first pass compute d2xdt2 directy from operator.
    if (nstate == 0) {
      f->Mult(x, dxdt, d2xdt2);
      nstate = 1;
    }

    // for (const int& d_id : *dirichlet_dofs_) {
    //   d2xdt2[d_id] = 0.0;
    // }

    // Predict alpha levels
    if (fixed_point_predict_alpha_level_) {
      add(dxdt, fac0_ * dt, d2xdt2, va);
      add(x, fac1_ * dt, va, xa);
      add(dxdt, fac2_ * dt, d2xdt2, va);
      fixed_point_predict_alpha_level_ = false;
    }

    // Solve alpha levels
    mimi_operator_->dt_ = dt;
    f->SetTime(t + dt);
    f->ImplicitSolve(fac3_ * dt * dt, fac4_ * dt, xa, va, aa);
  }

  virtual void FixedPointAdvance2(mfem::Vector& x,
                                  mfem::Vector& dxdt,
                                  double& t,
                                  double& dt) {
    MIMI_FUNC()
    if (fixed_point_predict_alpha_level_) {
      mimi::utils::PrintAndThrowError(
          "FixedPointAdvance2() should be called after FixedPointSolve2()");
    }

    // // xa and va are always freshly overwritten in fixedpointsolve,
    // // but would duplicate in AdvanceTime2, so assign a temp vector
    // fixed_point_tmp_xa_.SetSize(x.Size());
    // fixed_point_tmp_va_.SetSize(dxdt.Size());

    const double fac3dtdt = fac3_ * dt * dt;
    const double fac4dt = fac4_ * dt;
    // // Correct alpha levels
    // // xa.Add(fac3_ * dt * dt, aa); // <- do this
    // add(xa, fac3dtdt, Base_::aa, fixed_point_tmp_xa_);
    // // va.Add(fac4_ * dt, aa); // <- do this
    // add(va, fac4dt, Base_::aa, fixed_point_tmp_va_);

    // // extrapolate using temp vectors
    // x *= 1.0 - 1.0 / fac1_;
    // x.Add(1.0 / fac1_, fixed_point_tmp_xa_);

    // dxdt *= 1.0 - 1.0 / fac1_;
    // dxdt.Add(1.0 / fac1_, fixed_point_tmp_va_);

    const double prev_fac = 1. - fac1_inv_;
    double* x_ptr = x.GetData();
    double* v_ptr = dxdt.GetData();
    const double* xa_ptr = xa.GetData();
    const double* va_ptr = va.GetData();
    const double* aa_ptr = aa.GetData();
    for (int i{}; i < x.Size(); ++i) {
      const double aa_val = aa_ptr[i];
      x_ptr[i] =
          (x_ptr[i] * prev_fac) + (fac1_inv_ * (xa_ptr[i] + fac3dtdt * aa_val));
      v_ptr[i] =
          (v_ptr[i] * prev_fac) + (fac1_inv_ * (va_ptr[i] + fac4dt * aa_val));
    }
  }

  virtual void
  AdvanceTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
    MIMI_FUNC()

    // // Correct alpha levels
    // xa.Add(fac3_ * dt * dt, aa);
    // va.Add(fac4_ * dt, aa);

    // // Extrapolate
    // x *= 1.0 - 1.0 / fac1_;
    // x.Add(1.0 / fac1_, xa);

    // dxdt *= 1.0 - 1.0 / fac1_;
    // dxdt.Add(1.0 / fac1_, va);

    // d2xdt2 *= 1.0 - 1.0 / fac5_;
    // d2xdt2.Add(1.0 / fac5_, aa);

    // do what's above in one loop
    const double prev_fac = 1. - fac1_inv_;
    const double fac3dtdt = fac3_ * dt * dt;
    const double fac4dt = fac4_ * dt;
    double* x_p = x.GetData();
    double* v_p = dxdt.GetData();
    double* a_p = d2xdt2.GetData();
    double* xa_p = xa.GetData();
    double* va_p = va.GetData();
    const double* aa_p = aa.GetData();
    for (int i{}; i < x.Size(); ++i) {
      const double aa_val = aa_p[i];
      // corect alpha values
      xa_p[i] += fac3dtdt * aa_val;
      va_p[i] += fac4dt * aa_val;
      // extrapolate
      x_p[i] = (x_p[i] * prev_fac) + (fac1_inv_ * xa_p[i]);
      v_p[i] = (v_p[i] * prev_fac) + (fac1_inv_ * va_p[i]);
      a_p[i] = (a_p[i] * prev_fac) + (fac5_inv_ * aa_val);
    }

    t += dt;

    // now xa and va should be changing
    fixed_point_predict_alpha_level_ = true;
  }
};

/// The classical midpoint method.
class AverageAcceleration : public GeneralizedAlpha2 {
public:
  AverageAcceleration(mfem::SecondOrderTimeDependentOperator& oper) {
    alpha_m = 0.5;
    alpha_f = 0.5;
    beta = 0.25;
    gamma = 0.5;
    Init(oper);
    ComputeFactors();
    SaveMimiDownCast2(&oper);
  }
  virtual std::string Name() const { return "AverageAcceleration"; }
};

/// HHT-alpha ODE solver
/// Improved numerical dissipation for time integration algorithms
/// in structural dynamics
/// H.M. Hilber, T.J.R. Hughes and R.L. Taylor 1977
/// https://doi.org/10.1002/eqe.4290050306
/// alpha in [2/3,1] --> Defined differently than in paper.
class HHTAlpha : public GeneralizedAlpha2 {
public:
  using Base_ = GeneralizedAlpha2;

  HHTAlpha(mfem::SecondOrderTimeDependentOperator& oper, double alpha = 1.) {
    alpha = (alpha > 1.0) ? 1.0 : alpha;
    alpha = (alpha < 2.0 / 3.0) ? 2.0 / 3.0 : alpha;

    alpha_m = 1.0;
    alpha_f = alpha;
    beta = (2 - alpha) * (2 - alpha) / 4;
    gamma = 0.5 + alpha_m - alpha_f;

    Init(oper);
    ComputeFactors();
    SaveMimiDownCast2(&oper);
  }
  virtual std::string Name() const { return "HHTAlpha"; }
};

/// WBZ-alpha ODE solver
/// An alpha modification of Newmark's method
/// W.L. Wood, M. Bossak and O.C. Zienkiewicz 1980
/// https://doi.org/10.1002/nme.1620151011
/// rho_inf in [0,1]
class WBZAlpha : public GeneralizedAlpha2 {
public:
  WBZAlpha(mfem::SecondOrderTimeDependentOperator& oper, double rho_inf = 1.0) {
    rho_inf = (rho_inf > 1.0) ? 1.0 : rho_inf;
    rho_inf = (rho_inf < 0.0) ? 0.0 : rho_inf;

    alpha_f = 1.0;
    alpha_m = 2.0 / (1.0 + rho_inf);
    beta = 0.25 * pow(1.0 + alpha_m - alpha_f, 2);
    gamma = 0.5 + alpha_m - alpha_f;
    Init(oper);
    ComputeFactors();
  }
  virtual std::string Name() const { return "WBZAlpha"; }
};

// Imported the parts of Newmark from MFEM manually

/// The classical newmark method.
/// Newmark, N. M. (1959) A method of computation for structural dynamics.
/// Journal of Engineering Mechanics, ASCE, 85 (EM3) 67-94.
class Newmark : public mfem::SecondOrderODESolver, public OdeBase {
protected:
  mfem::Vector d2xdt2, xn, vn;
  double beta_, gamma_;
  double fac0_, fac1_, fac2_, fac3_, fac4_;
  bool first;

public:
  Newmark(mfem::SecondOrderTimeDependentOperator& oper,
          double beta = 0.25,
          double gamma = 0.5) {

    Init(oper);
    beta_ = beta;
    gamma_ = gamma;
    ComputeFactors();
    SaveMimiDownCast2(&oper);
  }

  virtual void ComputeFactors() {
    fac0_ = 0.5 - beta_;
    fac2_ = 1.0 - gamma_;
    fac3_ = beta_;
    fac4_ = gamma_;
  }

  virtual void Init(mfem::SecondOrderTimeDependentOperator& f_) {
    mfem::SecondOrderODESolver::Init(f_);
    d2xdt2.SetSize(f->Width());
    d2xdt2 = 0.0;
    xn.SetSize(f->Width());
    xn = 0.0;
    vn.SetSize(f->Width());
    vn = 0.0;
    first = true;
  }

  virtual void PrintProperties(std::ostream& os) {
    os << "Newmark time integrator:" << std::endl;
    os << "beta    = " << beta_ << std::endl;
    os << "gamma   = " << gamma_ << std::endl;

    if (gamma_ == 0.5) {
      os << "Second order"
         << " and ";
    } else {
      os << "First order"
         << " and ";
    }

    if ((gamma_ >= 0.5) && (beta_ >= (gamma_ + 0.5) * (gamma_ + 0.5) / 4)) {
      os << "A-Stable" << std::endl;
    } else if ((gamma_ >= 0.5) && (beta_ >= 0.5 * gamma_)) {
      os << "Conditionally stable" << std::endl;
    } else {
      os << "Unstable" << std::endl;
    }
  }

  virtual void
  Step(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {

    // In the first pass compute d2xdt2 directly from operator.
    if (first) {
      f->Mult(x, dxdt, d2xdt2);
      first = false;
    }
    mimi_operator_->dt_ = dt;
    f->SetTime(t + dt);

    x.Add(dt, dxdt);
    x.Add(fac0_ * dt * dt, d2xdt2);
    dxdt.Add(fac2_ * dt, d2xdt2);

    f->SetTime(t + dt);
    f->ImplicitSolve(fac3_ * dt * dt, fac4_ * dt, x, dxdt, d2xdt2);

    x.Add(fac3_ * dt * dt, d2xdt2);
    dxdt.Add(fac4_ * dt, d2xdt2);
    t += dt;
  }

  virtual std::string Name() const { return "Newmark"; }

  virtual void PrintInfo() {
    MIMI_FUNC()

    mimi::utils::PrintInfo("Info for", Name());
    PrintProperties(std::cout);
  }

  virtual void
  StepTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
    MIMI_FUNC()

    mimi_operator_->dt_ = dt;
    Step(x, dxdt, t, dt);
  }

  virtual void FixedPointSolve2(const mfem::Vector& x,
                                const mfem::Vector& dxdt,
                                double& t,
                                double& dt) {
    MIMI_FUNC()

    // In the first pass compute d2xdt2 directly from operator.
    if (first) {
      f->Mult(x, dxdt, d2xdt2);
      first = false;
    }
    mimi_operator_->dt_ = dt;
    f->SetTime(t + dt);

    add(x, dt, dxdt, xn);
    xn.Add(fac0_ * dt * dt, d2xdt2);
    add(dxdt, fac2_ * dt, d2xdt2, vn);

    f->SetTime(t + dt);
    f->ImplicitSolve(fac3_ * dt * dt, fac4_ * dt, xn, vn, d2xdt2);
  }

  virtual void FixedPointAdvance2(mfem::Vector& x,
                                  mfem::Vector& dxdt,
                                  double& t,
                                  double& dt) {
    MIMI_FUNC()

    x.Add(fac3_ * dt * dt, d2xdt2);
    dxdt.Add(fac4_ * dt, d2xdt2);
  }

  virtual void
  AdvanceTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
    MIMI_FUNC()

    add(xn, fac3_ * dt * dt, d2xdt2, x);
    add(vn, fac4_ * dt, d2xdt2, dxdt);

    t += dt;
  }
};

class LinearAcceleration : public Newmark {
public:
  LinearAcceleration(mfem::SecondOrderTimeDependentOperator& oper)
      : Newmark(oper, 1.0 / 6.0, 0.5) {}

  virtual std::string Name() const { return "LinearAcceleration"; }
};

class CentralDifference : public Newmark {
public:
  CentralDifference(mfem::SecondOrderTimeDependentOperator& oper)
      : Newmark(oper, 0.0, 0.5) {}
  virtual std::string Name() const { return "CentralDifference"; }
};

class FoxGoodwin : public Newmark {
public:
  FoxGoodwin(mfem::SecondOrderTimeDependentOperator& oper)
      : Newmark(oper, 1.0 / 12.0, 0.5) {}
  virtual std::string Name() const { return "FoxGoodwin"; }
};

} // namespace mimi::solvers
