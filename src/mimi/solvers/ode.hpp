#pragma once

#include <string>

#include <mfem.hpp>

#include "mimi/utils/print.hpp"

namespace mimi::solvers {

class OdeBase {
public:
  virtual ~OdeBase() = default;
  virtual std::string Name() const = 0;
  virtual void PrintInfo() {
    mimi::utils::PrintInfo("No detailed info for", Name());
  }

  virtual void
  StepTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("StepTime2 is not implemented for", Name());
  }

  virtual void
  FixedPointSolve2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
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
  double fac0_, fac1_, fac2_, fac3_, fac4_, fac5_;

public:
  using Base_ = mfem::GeneralizedAlpha2Solver;

  GeneralizedAlpha2() = default;

  GeneralizedAlpha2(mfem::SecondOrderTimeDependentOperator& oper,
                    double rho_inf = 0.5)
      : Base_(rho_inf) {

    Base_::Init(oper);
    ComputeFactors();
  }

  virtual void ComputeFactors() {
    fac0_ = (0.5 - (beta / alpha_m));
    fac1_ = alpha_f;
    fac2_ = alpha_f * (1.0 - (gamma / alpha_m));
    fac3_ = beta * alpha_f / alpha_m;
    fac4_ = gamma * alpha_f / alpha_m;
    fac5_ = alpha_m;
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

    Base_::Step(x, dxdt, t, dt);
  }
  virtual void
  FixedPointSolve2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
    MIMI_FUNC()

    // In the first pass compute d2xdt2 directy from operator.
    if (nstate == 0) {
      f->Mult(x, dxdt, d2xdt2);
      nstate = 1;
    }

    // Predict alpha levels
    add(dxdt, fac0_ * dt, d2xdt2, va);
    add(x, fac1_ * dt, va, xa);
    add(dxdt, fac2_ * dt, d2xdt2, va);

    // Solve alpha levels
    f->SetTime(t + dt);
    f->ImplicitSolve(fac3_ * dt * dt, fac4_ * dt, xa, va, aa);
  }

  virtual void FixedPointAdvance2(mfem::Vector& x,
                                  mfem::Vector& dxdt,
                                  double& t,
                                  double& dt) {
    MIMI_FUNC()

    // xa and va are always freshly overwritten in fixedpointsolve,
    // but would duplicate in AdvanceTime2, so assign a temp vector
    mfem::Vector tmp_xa(x.Size());
    mfem::Vector tmp_va(dxdt.Size());

    // Correct alpha levels
    // xa.Add(fac3_ * dt * dt, aa); // <- do this
    add(xa, fac3_ * dt * dt, Base_::aa, tmp_xa);
    // va.Add(fac4_ * dt, aa); // <- do this
    add(va, fac4_ * dt, Base_::aa, tmp_va);

    // extrapolate using temp vectors
    x *= 1.0 - 1.0 / fac1_;
    x.Add(1.0 / fac1_, tmp_xa);

    dxdt *= 1.0 - 1.0 / fac1_;
    dxdt.Add(1.0 / fac1_, tmp_va);
  }

  virtual void
  AdvanceTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
    MIMI_FUNC()

    // Correct alpha levels
    xa.Add(fac3_ * dt * dt, aa);
    va.Add(fac4_ * dt, aa);

    // Extrapolate
    x *= 1.0 - 1.0 / fac1_;
    x.Add(1.0 / fac1_, xa);

    dxdt *= 1.0 - 1.0 / fac1_;
    dxdt.Add(1.0 / fac1_, va);

    d2xdt2 *= 1.0 - 1.0 / fac5_;
    d2xdt2.Add(1.0 / fac5_, aa);

    t += dt;
  }
};

/// The classical midpoint method.
class AverageAccelerationSolver : public GeneralizedAlpha2 {
public:
  AverageAccelerationSolver(mfem::SecondOrderTimeDependentOperator& oper) {
    alpha_m = 0.5;
    alpha_f = 0.5;
    beta = 0.25;
    gamma = 0.5;
    ComputeFactors();
    Init(oper);
  }
  virtual std::string Name() const { return "AverageAcceleration"; }
};

/// HHT-alpha ODE solver
/// Improved numerical dissipation for time integration algorithms
/// in structural dynamics
/// H.M. Hilber, T.J.R. Hughes and R.L. Taylor 1977
/// https://doi.org/10.1002/eqe.4290050306
/// alpha in [2/3,1] --> Defined differently than in paper.
class HHTAlphaSolver : public GeneralizedAlpha2 {
public:
  HHTAlphaSolver(mfem::SecondOrderTimeDependentOperator& oper,
                 double alpha = 1.0) {
    alpha = (alpha > 1.0) ? 1.0 : alpha;
    alpha = (alpha < 2.0 / 3.0) ? 2.0 / 3.0 : alpha;

    alpha_m = 1.0;
    alpha_f = alpha;
    beta = (2 - alpha) * (2 - alpha) / 4;
    gamma = 0.5 + alpha_m - alpha_f;

    ComputeFactors();
    Init(oper);
  }
  virtual std::string Name() const { return "HHTAlpha"; }
};

/// WBZ-alpha ODE solver
/// An alpha modification of Newmark's method
/// W.L. Wood, M. Bossak and O.C. Zienkiewicz 1980
/// https://doi.org/10.1002/nme.1620151011
/// rho_inf in [0,1]
class WBZAlphaSolver : public GeneralizedAlpha2 {
public:
  WBZAlphaSolver(mfem::SecondOrderTimeDependentOperator& oper,
                 double rho_inf = 1.0) {
    rho_inf = (rho_inf > 1.0) ? 1.0 : rho_inf;
    rho_inf = (rho_inf < 0.0) ? 0.0 : rho_inf;

    alpha_f = 1.0;
    alpha_m = 2.0 / (1.0 + rho_inf);
    beta = 0.25 * pow(1.0 + alpha_m - alpha_f, 2);
    gamma = 0.5 + alpha_m - alpha_f;

    ComputeFactors();
    Init(oper);
  }
  virtual std::string Name() const { return "WBZAlpha"; }
};

// Imported the parts of Newmark from MFEM manually

/// The classical newmark method.
/// Newmark, N. M. (1959) A method of computation for structural dynamics.
/// Journal of Engineering Mechanics, ASCE, 85 (EM3) 67-94.
class NewmarkSolver : public mfem::SecondOrderODESolver, public OdeBase {
protected:
  mfem::Vector d2xdt2, xn, vn;
  double beta_, gamma_;
  double fac0_, fac1_, fac2_, fac3_, fac4_;
  bool first;

public:
  NewmarkSolver(mfem::SecondOrderTimeDependentOperator& oper,
                double beta = 0.25,
                double gamma = 0.5) {

    Init(oper);
    beta_ = beta;
    gamma_ = gamma;
    ComputeFactors();
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

    Step(x, dxdt, t, dt);
  }

  virtual void
  FixedPointSolve2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {
    MIMI_FUNC()

    // In the first pass compute d2xdt2 directly from operator.
    if (first) {
      f->Mult(x, dxdt, d2xdt2);
      first = false;
    }
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

class LinearAccelerationSolver : public NewmarkSolver {
public:
  LinearAccelerationSolver(mfem::SecondOrderTimeDependentOperator& oper)
      : NewmarkSolver(oper, 1.0 / 6.0, 0.5) {}

  virtual std::string Name() const { return "LinearAcceleration"; }
};

class CentralDifferenceSolver : public NewmarkSolver {
public:
  CentralDifferenceSolver(mfem::SecondOrderTimeDependentOperator& oper)
      : NewmarkSolver(oper, 0.0, 0.5) {}
  virtual std::string Name() const { return "CentralDifference"; }
};

class FoxGoodwinSolver : public NewmarkSolver {
public:
  FoxGoodwinSolver(mfem::SecondOrderTimeDependentOperator& oper)
      : NewmarkSolver(oper, 1.0 / 12.0, 0.5) {}
  virtual std::string Name() const { return "FoxGoodwin"; }
};

} // namespace mimi::solvers
