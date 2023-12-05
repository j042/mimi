#pragma once

#include <string>

#include <mfem.hpp>

#include "mimi/utils/print.hpp"

namespace mimi::solvers {

class OdeBase {
public:
  virtual ~OdeBase() = default;
  virtual std::string Name() = 0;
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

  GeneralizedAlpha2(mfem::SecondOrderTimeDependentOperator& oper,
                    double rho_inf = 0.5)
      : Base_(rho_inf) {
    Base_::Init(oper);

    // compute factors
    fac0_ = (0.5 - (beta / alpha_m));
    fac1_ = alpha_f;
    fac2_ = alpha_f * (1.0 - (gamma / alpha_m));
    fac3_ = beta * alpha_f / alpha_m;
    fac4_ = gamma * alpha_f / alpha_m;
    fac5_ = alpha_m;
  }

  virtual std::string Name() { return "GeneralizedAlpha2"; }

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

} // namespace mimi::solvers
