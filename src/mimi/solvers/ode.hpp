#pragma once

#include <string>

#include <mfem.hpp>

#include "mimi/operators/base.hpp"
#include "mimi/utils/boundary_conditions.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::solvers {

class OdeBase {
protected:
  // we keep casted mimi operator base to set dt_
  mimi::operators::OperatorBase* mimi_operator_;
  const mfem::Array<int>* dirichlet_dofs_{nullptr};

public:
  std::shared_ptr<mimi::utils::TimeDependentDirichletBoundaryCondition>
      dynamic_dirichlet_;

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

  virtual void ComputeFactors();

  virtual std::string Name() const { return "GeneralizedAlpha2"; }

  virtual void PrintInfo() {
    MIMI_FUNC()

    mimi::utils::PrintInfo("Info for", Name());
    Base_::PrintProperties();
  }

  virtual void
  StepTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt);

  virtual void FixedPointSolve2(const mfem::Vector& x,
                                const mfem::Vector& dxdt,
                                double& t,
                                double& dt);

  virtual void FixedPointAdvance2(mfem::Vector& x,
                                  mfem::Vector& dxdt,
                                  double& t,
                                  double& dt);

  virtual void
  AdvanceTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt);
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

  virtual void ComputeFactors();

  virtual void Init(mfem::SecondOrderTimeDependentOperator& f_);

  virtual void PrintProperties(std::ostream& os);

  virtual void Step(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt);

  virtual std::string Name() const { return "Newmark"; }

  virtual void PrintInfo() {
    MIMI_FUNC()

    mimi::utils::PrintInfo("Info for", Name());
    PrintProperties(std::cout);
  }

  virtual void
  StepTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt);

  virtual void FixedPointSolve2(const mfem::Vector& x,
                                const mfem::Vector& dxdt,
                                double& t,
                                double& dt);

  virtual void FixedPointAdvance2(mfem::Vector& x,
                                  mfem::Vector& dxdt,
                                  double& t,
                                  double& dt);

  virtual void
  AdvanceTime2(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt);
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
