#pragma once

#include <string>

#include <mfem.hpp>

#include "mimi/utils/print.hpp"

namespace mimi::solvers {

// GENERAL 

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

// SOLVER 1

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

// SOLVER 2

/// The classical newmark method.
/// Newmark, N. M. (1959) A method of computation for structural dynamics.
/// Journal of Engineering Mechanics, ASCE, 85 (EM3) 67-94.
class NewmarkSolver : public mfem::SecondOrderODESolver
{
protected:
   Vector d2xdt2;

   double beta, gamma;
   bool first;

public:
   NewmarkSolver(double beta_ = 0.25, double gamma_ = 0.5) { beta = beta_; gamma = gamma_; };

   void PrintProperties(std::ostream &out = mfem::out);

   void Init(mfem::SecondOrderTimeDependentOperator &f_) override;

   void Step(mfem::Vector &x, mfem::Vector &dxdt, double &t, double &dt) override;
};

/*
class LinearAccelerationSolver : public NewmarkSolver
{
public:
   LinearAccelerationSolver() : NewmarkSolver(1.0/6.0, 0.5) { };
};

class CentralDifferenceSolver : public NewmarkSolver
{
public:
   CentralDifferenceSolver() : NewmarkSolver(0.0, 0.5) { };
};

class FoxGoodwinSolver : public NewmarkSolver
{
public:
   FoxGoodwinSolver() : NewmarkSolver(1.0/12.0, 0.5) { };
};
*/

/// TEIL AUS MFEM ode.cpp
void NewmarkSolver::Init(SecondOrderTimeDependentOperator &f_)
{
   SecondOrderODESolver::Init(f_);
   d2xdt2.SetSize(f->Width());
   d2xdt2 = 0.0;
   first = true;
}

void NewmarkSolver::PrintProperties(std::ostream &os)
{
   os << "Newmark time integrator:" << std::endl;
   os << "beta    = " << beta  << std::endl;
   os << "gamma   = " << gamma << std::endl;

   if (gamma == 0.5)
   {
      os<<"Second order"<<" and ";
   }
   else
   {
      os<<"First order"<<" and ";
   }

   if ((gamma >= 0.5) && (beta >= (gamma + 0.5)*(gamma + 0.5)/4))
   {
      os<<"A-Stable"<<std::endl;
   }
   else if ((gamma >= 0.5) && (beta >= 0.5*gamma))
   {
      os<<"Conditionally stable"<<std::endl;
   }
   else
   {
      os<<"Unstable"<<std::endl;
   }
}

void NewmarkSolver::Step(Vector &x, Vector &dxdt, double &t, double &dt)
{
   double fac0 = 0.5 - beta;
   double fac2 = 1.0 - gamma;
   double fac3 = beta;
   double fac4 = gamma;

   // In the first pass compute d2xdt2 directly from operator.
   if (first)
   {
      f->Mult(x, dxdt, d2xdt2);
      first = false;
   }
   f->SetTime(t + dt);

   x.Add(dt, dxdt);
   x.Add(fac0*dt*dt, d2xdt2);
   dxdt.Add(fac2*dt, d2xdt2);

   f->SetTime(t + dt);
   f->ImplicitSolve(fac3*dt*dt, fac4*dt, x, dxdt, d2xdt2);

   x   .Add(fac3*dt*dt, d2xdt2);
   dxdt.Add(fac4*dt,    d2xdt2);
   t += dt;
}
/// TEIL ENDE


class Newmark : public NewmarkSolver, public OdeBase {
protected:
  double fac0_, fac1_, fac2_, fac3_, fac4_;

public:
  using Base_ = NewmarkSolver;

  Newmark(mfem::SecondOrderTimeDependentOperator& oper,
                    double beta = 0.25, double gamma = 0.5)
      : Base_(beta, gamma) {
    Base_::Init(oper);

    // compute factors
    fac0_ = 0.5 - beta;
    fac2_ = 1.0 - gamma;
    fac3_ = beta;
    fac4_ = gamma;
  }

  virtual std::string Name() { return "Newmark"; }

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

    // In the first pass, compute d2xdt2 directly from operator.
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
