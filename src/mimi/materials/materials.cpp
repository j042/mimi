#include "mimi/materials/materials.hpp"

#include <algorithm>
#include <cmath>

#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/materials/material_hardening.hpp"
#include "mimi/materials/material_state.hpp"
#include "mimi/materials/material_utils.hpp"
#include "mimi/solvers/newton.hpp"
#include "mimi/utils/ad.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::materials {

void MaterialBase::SetYoungPoisson(const double young, const double poisson) {
  young_ = young;
  poisson_ = poisson;
  lambda_ = young * poisson / ((1 + poisson) * (1 - 2 * poisson));
  mu_ = young / (2.0 * (1.0 + poisson));
  G_ = mu_;
  K_ = young / (3.0 * (1.0 - (2.0 * poisson)));
}

void MaterialBase::SetLame(const double lambda, const double mu) {
  young_ = mu * (3 * lambda + 2 * mu) / (lambda + mu);
  poisson_ = lambda / (2 * (lambda + mu));
  lambda_ = lambda;
  mu_ = mu;
  G_ = mu;
  K_ = lambda + 2 * mu / 3;
}

void MaterialBase::Setup(const int dim) {
  MIMI_FUNC()

  dim_ = dim;

  mimi::utils::PrintInfo(Name(),
                         "Material Paramters:",
                         "\nE:",
                         young_,
                         "\npoisson:",
                         poisson_,
                         "\nlambda:",
                         lambda_,
                         "\nmu",
                         mu_,
                         "\nG",
                         G_,
                         "\nK",
                         K_);
}

void MaterialBase::EvaluateCauchy(const MaterialStatePtr_& state,
                                  WorkData_& tmp,
                                  mfem::DenseMatrix& sigma) const {
  MIMI_FUNC()

  // get P
  EvaluatePK1(state, tmp, tmp.alternative_stress_);

  // 1 / det(F) * P * F^T
  // they don't have mfem::Mult_a_ABt();
  mfem::MultABt(tmp.alternative_stress_, tmp.F_, sigma);
  sigma *= 1. / tmp.DetF();
}

void MaterialBase::EvaluatePK1(const MaterialStatePtr_& state,
                               WorkData_& tmp,
                               mfem::DenseMatrix& P) const {
  MIMI_FUNC()

  // get sigma
  EvaluateCauchy(state, tmp, tmp.alternative_stress_);

  // P = det(F) * sigma * F^-T
  mfem::MultABt(tmp.alternative_stress_, tmp.FInv(), P);
  P *= tmp.DetF();
}

void StVenantKirchhoff::EvaluatePK1(const MaterialStatePtr_& state,
                                    WorkData_& tmp,
                                    mfem::DenseMatrix& P) const {
  MIMI_FUNC()

  // get aux
  mfem::DenseMatrix& C = tmp.aux_mat_[k_C];
  mfem::DenseMatrix& E = tmp.aux_mat_[k_E];
  mfem::DenseMatrix& S = tmp.aux_mat_[k_S];

  // C
  mfem::MultAtB(tmp.F_, tmp.F_, C);

  // E
  mfem::Add(.5, C, -.5, tmp.I_, E);

  // S
  mfem::Add(lambda_ * E.Trace(), tmp.I_, 2 * mu_, E, S);

  // P
  mfem::Mult(tmp.F_, S, P);
}

void CompressibleOgdenNeoHookean::EvaluateCauchy(
    const MaterialStatePtr_& state,
    WorkData_& tmp,
    mfem::DenseMatrix& sigma) const {
  MIMI_FUNC()

  // get aux
  mfem::DenseMatrix& B = tmp.aux_mat_[k_B];

  // precompute aux values
  const double det_F = tmp.DetF();
  const double mu_over_det_F = mu_ / det_F;
  mfem::MultABt(tmp.F_, tmp.F_, B); // left green

  // flattening the eq above,
  // mu / det(F) B - mu / det(F) I + lambda * (det(F) - 1) I
  // mu / det(F) B + (-mu / det(F) + lambda * (det(F) - 1)) I
  mfem::Add(mu_over_det_F,
            B,
            -mu_over_det_F + lambda_ * (det_F - 1.),
            tmp.I_,
            sigma);
}

J2Linear::MaterialStatePtr_ J2Linear::CreateState() const {
  MIMI_FUNC();

  std::shared_ptr<State> state = std::make_shared<State>();
  // create 2 matrices with the size of dim x dim and zero initialize
  state->matrices.resize(state->k_state_matrices);
  for (mfem::DenseMatrix& mat : state->matrices) {
    mat.SetSize(dim_, dim_);
    mat = 0.;
  }
  // one scalar, also zero
  state->scalars.resize(state->k_state_scalars, 0.);
  return state;
}

void J2::Setup(const int dim) {
  MIMI_FUNC()

  /// base setup for conversions and dim, nthread
  Base_::Setup(dim);

  if (hardening_) {
    hardening_->Validate();
    // this does nothing for temperature independent hardening
    hardening_->InitializeTemperature(initial_temperature_,
                                      melting_temperature_);
  } else {
    mimi::utils::PrintAndThrowError("hardening missing for", Name());
  }
}

J2::MaterialStatePtr_ J2::CreateState() const {
  MIMI_FUNC();

  std::shared_ptr<State> state = std::make_shared<State>();
  // create 2 matrices with the size of dim x dim and zero initialize
  state->matrices.resize(state->k_state_matrices);
  for (mfem::DenseMatrix& mat : state->matrices) {
    mat.SetSize(dim_, dim_);
    mat = 0.;
  }
  // two scalars
  state->scalars.resize(state->k_state_scalars, 0.);
  state->scalars[State::k_temperature] = initial_temperature_;

  return state;
}

void J2Simo::Setup(const int dim) {
  MIMI_FUNC()

  /// base setup for conversions and dim, nthread
  Base_::Setup(dim);

  if (hardening_) {
    hardening_->Validate();
    // this does nothing for temperature independent hardening
    hardening_->InitializeTemperature(initial_temperature_,
                                      melting_temperature_);
  } else {
    mimi::utils::PrintAndThrowError("hardening missing for", Name());
  }
}

J2Simo::MaterialStatePtr_ J2Simo::CreateState() const {
  MIMI_FUNC();

  std::shared_ptr<State> state = std::make_shared<State>();
  // create 2 matrices with the size of dim x dim and zero initialize
  state->matrices.resize(state->k_state_matrices);
  for (mfem::DenseMatrix& mat : state->matrices) {
    mat.SetSize(dim_, dim_);
    mat = 0.;
  }
  state->matrices[State::k_be_old].Diag(1., dim_);
  state->matrices[State::k_F_old].Diag(1., dim_);

  // one scalar, also zero
  state->scalars.assign(state->k_state_scalars, 0.);

  // set initial temp
  state->scalars[State::k_temperature] = initial_temperature_;
  return state;
}

void J2Log::Setup(const int dim) {
  MIMI_FUNC()

  /// base setup for conversions and dim, nthread
  Base_::Setup(dim);

  if (hardening_) {
    hardening_->Validate();
    // this does nothing for temperature independent hardening
    hardening_->InitializeTemperature(initial_temperature_,
                                      melting_temperature_);
  } else {
    mimi::utils::PrintAndThrowError("hardening missing for", Name());
  }
}

J2Log::MaterialStatePtr_ J2Log::CreateState() const {
  MIMI_FUNC()

  std::shared_ptr<State> state = std::make_shared<State>();
  // create 2 matrices with the size of dim x dim and zero initialize
  state->matrices.resize(state->k_state_matrices);
  for (mfem::DenseMatrix& mat : state->matrices) {
    mat.SetSize(dim_, dim_);
    mat = 0.;
  }

  // for logarithmic this is I
  state->matrices[State::k_Fp_inv].Diag(1., dim_);

  // one scalar, also zero
  state->scalars.resize(state->k_state_scalars, 0.);

  // set initial temp
  state->scalars[State::k_temperature] = initial_temperature_;
  return state;
}

} // namespace mimi::materials
