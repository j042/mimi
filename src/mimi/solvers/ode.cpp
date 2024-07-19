#include "mimi/solvers/ode.hpp"

namespace mimi::solvers {

void GeneralizedAlpha2::ComputeFactors() {
  fac0_ = (0.5 - (beta / alpha_m));
  fac1_ = alpha_f;
  fac1_inv_ = 1. / fac1_;
  fac2_ = alpha_f * (1.0 - (gamma / alpha_m));
  fac3_ = beta * alpha_f / alpha_m;
  fac4_ = gamma * alpha_f / alpha_m;
  fac5_ = alpha_m;
  fac5_inv_ = 1. / fac5_;
}

void GeneralizedAlpha2::StepTime2(mfem::Vector& x,
                                  mfem::Vector& dxdt,
                                  double& t,
                                  double& dt) {
  MIMI_FUNC()

  mimi_operator_->dt_ = dt;
  // Base_::Step(x, dxdt, t, dt);

  const double prev_fac = 1. - fac1_inv_;
  const double fac0dt = fac0_ * dt;
  const double fac1dt = fac1_ * dt;
  const double fac2dt = fac2_ * dt;
  const double fac3dtdt = fac3_ * dt * dt;
  const double fac4dt = fac4_ * dt;
  double* x_p = x.GetData();
  double* v_p = dxdt.GetData();
  double* a_p = d2xdt2.GetData();
  double* xa_p = xa.GetData();
  double* va_p = va.GetData();
  const double* aa_p = aa.GetData();

  if (nstate == 0) {
    f->Mult(x, dxdt, d2xdt2);
    nstate = 1;
    aa = 0.0; // initialize aa as we want to use iterative mode for newton
  }

  // Predict alpha levels - same as the following part
  // add(dxdt, fac0 * dt, d2xdt2, va);
  // add(x, fac1 * dt, va, xa);
  // add(dxdt, fac2 * dt, d2xdt2, va);
  for (int i{}; i < x.Size(); ++i) {
    xa_p[i] = x_p[i] + (v_p[i] + fac0dt * a_p[i]) * fac1dt;
    va_p[i] = v_p[i] + fac2dt * a_p[i];
  }

  // solve alpha levels
  f->SetTime(t + dt);
  // apply BC
  if (dynamic_dirichlet_) {
    dynamic_dirichlet_->Apply(t, dt, x, dxdt, d2xdt2, xa, va, aa);
  }
  f->ImplicitSolve(fac3dtdt, fac4dt, xa, va, aa);

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

  // apply BC again
  if (dynamic_dirichlet_) {
    dynamic_dirichlet_->Restore(x, dxdt, d2xdt2);
  }

  mimi_operator_->PostTimeAdvance(x, dxdt);
  t += dt; // advance time within
}

void GeneralizedAlpha2::FixedPointSolve2(const mfem::Vector& x,
                                         const mfem::Vector& dxdt,
                                         double& t,
                                         double& dt) {
  MIMI_FUNC()

  // In the first pass compute d2xdt2 directy from operator.
  if (nstate == 0) {
    f->Mult(x, dxdt, d2xdt2);
    nstate = 1;
  }

  // Predict alpha levels
  if (fixed_point_predict_alpha_level_) {
    add(dxdt, fac0_ * dt, d2xdt2, va);
    add(x, fac1_ * dt, va, xa);
    add(dxdt, fac2_ * dt, d2xdt2, va);
    fixed_point_predict_alpha_level_ = false;
    // double check if aa is zero where it should be
    if (dynamic_dirichlet_) {
      dynamic_dirichlet_->Apply(t, dt, x, dxdt, d2xdt2, xa, va, aa);
    }
  }

  // Solve alpha levels
  mimi_operator_->dt_ = dt;

  // solve alpha levels
  f->SetTime(t + dt);
  f->ImplicitSolve(fac3_ * dt * dt, fac4_ * dt, xa, va, aa);
}

void GeneralizedAlpha2::FixedPointAdvance2(mfem::Vector& x,
                                           mfem::Vector& dxdt,
                                           double& t,
                                           double& dt) {
  MIMI_FUNC()
  if (fixed_point_predict_alpha_level_) {
    mimi::utils::PrintAndThrowError(
        "FixedPointAdvance2() should be called after FixedPointSolve2()");
  }

  // // xa and va are always freshly overwritten in fixedpointsolve,

  const double fac3dtdt = fac3_ * dt * dt;
  const double fac4dt = fac4_ * dt;
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

  // apply BC again
  if (dynamic_dirichlet_) {
    dynamic_dirichlet_->Restore(x, dxdt, d2xdt2);
  }
}

void GeneralizedAlpha2::AdvanceTime2(mfem::Vector& x,
                                     mfem::Vector& dxdt,
                                     double& t,
                                     double& dt) {
  MIMI_FUNC()

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

  // now xa and va should be changed
  fixed_point_predict_alpha_level_ = true;

  // apply BC again
  if (dynamic_dirichlet_) {
    dynamic_dirichlet_->Restore(x, dxdt, d2xdt2);
  }

  // accumulate!
  mimi_operator_->PostTimeAdvance(x, dxdt);
}

void Newmark::ComputeFactors() {
  fac0_ = 0.5 - beta_;
  fac2_ = 1.0 - gamma_;
  fac3_ = beta_;
  fac4_ = gamma_;
}

void Newmark::Init(mfem::SecondOrderTimeDependentOperator& f_) {
  mfem::SecondOrderODESolver::Init(f_);
  d2xdt2.SetSize(f->Width());
  d2xdt2 = 0.0;
  xn.SetSize(f->Width());
  xn = 0.0;
  vn.SetSize(f->Width());
  vn = 0.0;
  first = true;
}

void Newmark::PrintProperties(std::ostream& os) {
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

void Newmark::Step(mfem::Vector& x, mfem::Vector& dxdt, double& t, double& dt) {

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
  mimi_operator_->PostTimeAdvance(x, dxdt);
}

void Newmark::StepTime2(mfem::Vector& x,
                        mfem::Vector& dxdt,
                        double& t,
                        double& dt) {
  MIMI_FUNC()

  mimi_operator_->dt_ = dt;
  Step(x, dxdt, t, dt);
  mimi_operator_->PostTimeAdvance(x, dxdt);
}

void Newmark::FixedPointSolve2(const mfem::Vector& x,
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

void Newmark::FixedPointAdvance2(mfem::Vector& x,
                                 mfem::Vector& dxdt,
                                 double& t,
                                 double& dt) {
  MIMI_FUNC()

  x.Add(fac3_ * dt * dt, d2xdt2);
  dxdt.Add(fac4_ * dt, d2xdt2);
}

void Newmark::AdvanceTime2(mfem::Vector& x,
                           mfem::Vector& dxdt,
                           double& t,
                           double& dt) {
  MIMI_FUNC()

  add(xn, fac3_ * dt * dt, d2xdt2, x);
  add(vn, fac4_ * dt, d2xdt2, dxdt);

  t += dt;
  mimi_operator_->PostTimeAdvance(x, dxdt);
}
} // namespace mimi::solvers
