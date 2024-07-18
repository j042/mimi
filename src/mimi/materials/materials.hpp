#pragma once

#include <algorithm>
#include <cmath>

#include <mfem.hpp>

#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/materials/material_hardening.hpp"
#include "mimi/materials/material_state.hpp"
#include "mimi/materials/material_utils.hpp"
#include "mimi/solvers/newton.hpp"
#include "mimi/utils/ad.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::materials {

/// Material base.
/// We can either define material or material state (in this case one material)
/// at each quad point. Something to consider before implementing everything
class MaterialBase {
public:
  using MaterialStatePtr_ = std::shared_ptr<MaterialState>;
  using WorkData_ = mimi::integrators::NonlinearSolidWorkData;

  double dt_;
  double first_effective_dt_;
  double second_effective_dt_;

protected:
  int dim_;

public:
  double density_{-1.0};
  double viscosity_{-1.0};
  double lambda_{-1.0};
  double mu_{-1.0};
  double young_{-1.0};
  double poisson_{-1.0};
  double K_{-1.0};
  double G_{-1.0};

  virtual std::string Name() const { return "MaterialBase"; }

  /// defining material properties #1
  virtual void SetYoungPoisson(const double young, const double poisson) {
    young_ = young;
    poisson_ = poisson;
    lambda_ = young * poisson / ((1 + poisson) * (1 - 2 * poisson));
    mu_ = young / (2.0 * (1.0 + poisson));
    G_ = mu_;
    K_ = young / (3.0 * (1.0 - (2.0 * poisson)));
  }

  /// defining the material properties #2
  virtual void SetLame(const double lambda, const double mu) {
    young_ = mu * (3 * lambda + 2 * mu) / (lambda + mu);
    poisson_ = lambda / (2 * (lambda + mu));
    lambda_ = lambda;
    mu_ = mu;
    G_ = mu;
    K_ = lambda + 2 * mu / 3;
  }

  /// self setup. will be called once.
  /// unless you want to do it your self, call this one.
  /// before you extend Setup.
  virtual void Setup(const int dim) {
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

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("AllocateAux not implemented for", Name());
  }

  /// answers if this material is suitable for visco-solids.
  virtual bool IsRateDependent() const { return false; }

  /// each material that needs states can implement this.
  /// else, just nullptr
  virtual MaterialStatePtr_ CreateState() const { return MaterialStatePtr_{}; };

  /// @brief unless implemented, this will try to call evaluate PK1 and
  /// transform if none of stress is implemented, you will be stuck in a
  /// neverending loop current implementation is not so memory efficient
  /// @param F
  /// @param state
  /// @param sigma
  virtual void EvaluateCauchy(const MaterialStatePtr_& state,
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

  /// @brief unless implemented, this will try to call evaluate sigma and
  /// transform if none of stress is implemented, you will be stuck in a
  /// neverending loop current implementation is not so memory efficient
  virtual void EvaluatePK1(const MaterialStatePtr_& state,
                           WorkData_& tmp,
                           mfem::DenseMatrix& P) const {
    MIMI_FUNC()

    // get sigma
    EvaluateCauchy(state, tmp, tmp.alternative_stress_);

    // P = det(F) * sigma * F^-T
    mfem::MultABt(tmp.alternative_stress_, tmp.FInv(), P);
    P *= tmp.DetF();
  }

  /// state accumulating version
  virtual void Accumulate(MaterialStatePtr_& state, WorkData_& tmp) const {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("Accumulate() not implemented for", Name());
  }
};

class StVenantKirchhoff : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;

protected:
  /// just for lookup index
  static constexpr const int k_C{0};
  static constexpr const int k_E{1};
  static constexpr const int k_S{2};

public:
  virtual std::string Name() const { return "StVenantKirchhoff"; }

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    tmp.aux_mat_.resize(3, mfem::DenseMatrix(dim_, dim_));
  }

  virtual void EvaluatePK1(const MaterialStatePtr_& state,
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
};

/// @brief https://en.wikipedia.org/wiki/Neo-Hookean_solid
/// mu / det(F) * (B - I) + lambda * (det(F) - 1) I
/// B = FF^t
class CompressibleOgdenNeoHookean : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;

protected:
  /// just for lookup index
  static constexpr const int k_B{0};

public:
  virtual std::string Name() const { return "CompressibleOdgenNeoHookean"; }

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    tmp.aux_mat_.resize(1, mfem::DenseMatrix(dim_, dim_));
  }

  /// mu / det(F) * (B - I) + lambda * (det(F) - 1) I
  virtual void EvaluateCauchy(const MaterialStatePtr_& state,
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
};

/// @brief Computational Methods for plasticity p260, box 7.5
/// Implementation reference from serac
class J2Linear : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;

  // additional parameters
  double isotropic_hardening_;
  double kinematic_hardening_;
  double sigma_y_; // yield

  /// decided to use inline
  bool is_2d_{true};

  struct State : public MaterialState {
    static constexpr const int k_state_matrices{2};
    static constexpr const int k_state_scalars{2};
    /// matrix indices
    static constexpr const int k_plastic_strain{0};
    static constexpr const int k_beta{1};
    /// scalar indices
    static constexpr const int k_accumulated_plastic_strain{0};
  };

protected:
  /// some constants
  const double sqrt_6_ = std::sqrt(6.0);
  const double sqrt_3_2_ = std::sqrt(3.0 / 2.0);
  const double sqrt_2_3_ = std::sqrt(2.0 / 3.0);

  /// lookup index for matrix
  /// elastic strain
  static constexpr const int k_eps{0};
  /// stress deviator
  static constexpr const int k_s{1};
  /// elastic trial relative stress
  static constexpr const int k_eta{2};

public:
  virtual std::string Name() const { return "J2Linear"; }

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    tmp.aux_mat_.resize(3, mfem::DenseMatrix(dim_, dim_));
  }

  virtual MaterialStatePtr_ CreateState() const {
    MIMI_FUNC();

    std::shared_ptr<State> state = std::make_shared<State>();
    // create 2 matrices with the size of dim x dim and zero initialize
    state->matrices.resize(state->k_state_matrices);
    for (mfem::DenseMatrix& mat : state->matrices) {
      mat.SetSize(dim_, dim_);
      mat = 0.;
    }
    // one scalar, also zero
    state->scalars_.resize(state->k_state_scalars, 0.);
    return state;
  }

  template<bool accumulate>
  void PlasticStress(std::conditional_t<accumulate,
                                        MaterialStatePtr_,
                                        const MaterialStatePtr_>& state,
                     WorkData_& tmp,
                     mfem::DenseMatrix& sigma) const {
    // get aux
    mfem::DenseMatrix& eps = tmp.aux_mat_[k_eps];
    mfem::DenseMatrix& s = tmp.aux_mat_[k_s];
    mfem::DenseMatrix& eta = tmp.aux_mat_[k_eta];

    // get states
    auto& beta = state->matrices[State::k_beta];
    auto& plastic_strain = state->matrices[State::k_plastic_strain];
    auto& accumulated_plastic_strain =
        state->scalars_[State::k_accumulated_plastic_strain];

    // precompute aux values
    // eps, p, s, eta, q, phi
    ElasticStrain(tmp.F_, plastic_strain, eps);
    const double p = K_ * eps.Trace();
    Dev(eps, dim_, 2.0 * G_, s);
    mfem::Add(s, beta, -1., eta);
    const double eta_norm = Norm(eta);
    const double q = sqrt_3_2_ * eta_norm;
    const double phi =
        q - (sigma_y_ + isotropic_hardening_ * accumulated_plastic_strain);

    if (phi > 0.) {
      // 7.207
      const double plastic_strain_inc =
          phi / (3. * G_ + kinematic_hardening_ + isotropic_hardening_);
      // normalize eta in-place as we don't need it anymore
      eta *= 1. / eta_norm;
      if constexpr (!accumulate) {
        // return mapping
        s.Add(-sqrt_6_ * G_ * plastic_strain_inc, eta);
      }
      if constexpr (accumulate) { // this part should only be done at stepping.
        accumulated_plastic_strain += plastic_strain_inc;
        plastic_strain.Add(sqrt_3_2_ * plastic_strain_inc, eta);
        beta.Add(sqrt_2_3_ * kinematic_hardening_ * plastic_strain_inc, eta);
      }
    }

    // returning s + p * I
    if constexpr (!accumulate) {
      mfem::Add(s, tmp.I_, p, sigma);
    }
  }

  virtual void EvaluateCauchy(const MaterialStatePtr_& state,
                              WorkData_& tmp,
                              mfem::DenseMatrix& sigma) const {
    MIMI_FUNC()
    PlasticStress<false>(state, tmp, sigma);
  }

  virtual void Accumulate(MaterialStatePtr_& state, WorkData_& tmp) const {
    MIMI_FUNC()
    PlasticStress<true>(state, tmp, tmp.stress_ /* unused */);
  }
};

/// @brief Computational Methods for plasticity p260, box 7.5
/// This one excludes kinematic hardening,
/// which eliminates beta.
/// Then eta = s, instead of eta = s - beta
/// eta: relative stress
/// beta: backstress tensor
/// s: stress deviator
/// Implementation reference from serac
/// Considers nonlinear Isotropic hardening
class J2 : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;
  using HardeningPtr_ = std::shared_ptr<HardeningBase>;
  using ADScalar_ = typename HardeningBase::ADScalar_;

  // additional parameters
  HardeningPtr_ hardening_;

  static constexpr const double k_tol{1.e-10};

  struct State : public MaterialState {
    static constexpr const int k_state_matrices{1};
    static constexpr const int k_state_scalars{1};
    /// matrix indices
    static constexpr const int k_plastic_strain{0};
    /// scalar indices
    static constexpr const int k_accumulated_plastic_strain{0};
  };

protected:
  /// some constants
  const double sqrt_3_2_ = std::sqrt(3.0 / 2.0);
  /// elastic strain
  static constexpr const int k_eps{0};
  /// stress deviator
  static constexpr const int k_s{1};
  /// N_p - flow vector
  static constexpr const int k_N_p{2};

public:
  virtual std::string Name() const { return "J2NonlinearIsotropic"; }

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    tmp.aux_mat_.resize(3, mfem::DenseMatrix(dim_, dim_));
  }

  virtual MaterialStatePtr_ CreateState() const {
    MIMI_FUNC();

    std::shared_ptr<State> state = std::make_shared<State>();
    // create 2 matrices with the size of dim x dim and zero initialize
    state->matrices.resize(state->k_state_matrices);
    for (mfem::DenseMatrix& mat : state->matrices) {
      mat.SetSize(dim_, dim_);
      mat = 0.;
    }
    // one scalar, also zero
    state->scalars_.resize(state->k_state_scalars, 0.);
    return state;
  };

  template<bool accumulate>
  void PlasticStress(std::conditional_t<accumulate,
                                        MaterialStatePtr_,
                                        const MaterialStatePtr_>& state,
                     WorkData_& tmp,
                     mfem::DenseMatrix& sigma) const {
    MIMI_FUNC()

    mfem::DenseMatrix& eps = tmp.aux_mat_[k_eps];
    mfem::DenseMatrix& s = tmp.aux_mat_[k_s];
    mfem::DenseMatrix& N_p = tmp.aux_mat_[k_N_p];

    // get states - will get corresponding const-ness
    auto& plastic_strain = state->matrices[State::k_plastic_strain];
    auto& accumulated_plastic_strain =
        state->scalars_[State::k_accumulated_plastic_strain];

    // precompute aux values
    // eps, p, s, eta, q, phi
    ElasticStrain(tmp.F_, plastic_strain, eps);
    const double p = K_ * eps.Trace();
    Dev(eps, dim_, 2.0 * G_, s);
    const double q = sqrt_3_2_ * Norm(s);

    // admissibility
    const double eqps_old = accumulated_plastic_strain;
    auto residual = [eqps_old, q, *this](auto delta_eqps) -> ADScalar_ {
      return q - 3.0 * G_ * delta_eqps
             - hardening_->Evaluate(eqps_old + delta_eqps);
    };

    const double tolerance = hardening_->SigmaY() * k_tol;

    if (residual(0.0) > tolerance) {
      /// return mapping
      mimi::solvers::ScalarSolverOptions opts{.xtol = 0.,
                                              .rtol = tolerance,
                                              .max_iter = 100};

      const double lower_bound = 0.0;
      const double upper_bound =
          (q - hardening_->Evaluate(eqps_old).GetValue()) / (3.0 * G_);
      const double delta_eqps = mimi::solvers::ScalarSolve(residual,
                                                           0.0,
                                                           lower_bound,
                                                           upper_bound,
                                                           opts);
      // compute sqrt(3/2) * eta / norm(eta)
      // this term is use for both s and plastic strain
      // this is equivalent to
      // s = eta (see above; this is because we only consider isotropic)
      // 3/2 * s / q = 3/2 * s * sqrt(2/3) / norm(s)
      // but as this references serac's implementation
      // here it goes
      N_p.Set(1.5 / q, s);

      if constexpr (!accumulate) {
        s.Add(-2.0 * G_ * delta_eqps, N_p);
      }

      // no accumulation here
      if constexpr (accumulate) {
        accumulated_plastic_strain += delta_eqps;
        plastic_strain.Add(delta_eqps, N_p);
      }
    }

    // returning s + p * I
    if constexpr (!accumulate) {
      mfem::Add(s, tmp.I_, p, sigma);
    }
  }

  virtual void EvaluateCauchy(const MaterialStatePtr_& state,
                              WorkData_& tmp,
                              mfem::DenseMatrix& sigma) const {
    MIMI_FUNC()
    PlasticStress<false>(state, tmp, sigma);
  }

  virtual void Accumulate(MaterialStatePtr_& state, WorkData_& tmp) const {
    MIMI_FUNC()
    PlasticStress<true>(state, tmp, tmp.stress_ /* unused */);
  }
};

/// @brief Computational Methods for plasticity p260, box 7.5
/// specialized for thermo(adiabatic) visco hardening law
/// Note that this is an adiabatic process
/// Isotropic: eta = s, instead of eta = s - beta
/// eta: relative stress
/// beta: back stress
/// s: stress deviator
class J2AdiabaticVisco : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;
  using HardeningPtr_ = std::shared_ptr<HardeningBase>;
  using ADScalar_ = typename HardeningBase::ADScalar_;

  // additional parameters
  HardeningPtr_ hardening_;

  // thermo related
  double heat_fraction_{0.9};        // 0.9 is apparently abaqus' default
  double specific_heat_;             // for now, constant
  double initial_temperature_{20.0}; //
  double melting_temperature_{};     // takes from hardening - TODO - change

  static constexpr const double k_tol{1.e-10};

  struct State : public MaterialState {
    static constexpr const int k_state_matrices{2};
    static constexpr const int k_state_scalars{2};
    /// matrix indices
    static constexpr const int k_plastic_strain{0};
    static constexpr const int k_elastic_strain{1};
    /// scalar indices
    static constexpr const int k_accumulated_plastic_strain{0};
    static constexpr const int k_temperature{1};
  };

protected:
  /// some constants
  const double sqrt_3_2_ = std::sqrt(3.0 / 2.0);

  /// lookup index for matrix
  /// elastic strain
  static constexpr const int k_eps{0};
  /// stress deviator
  static constexpr const int k_s{1};
  /// N_p
  static constexpr const int k_N_p{2};
  /// optional velocity gradient
  static constexpr const int k_work0{3};

public:
  virtual std::string Name() const { return "J2AdiabaticVisco"; }

  /// answers if this material is suitable for visco-solids.
  virtual bool IsRateDependent() const { return true; }

  virtual void Setup(const int dim) {
    MIMI_FUNC()

    /// base setup for conversions and dim, nthread
    Base_::Setup(dim);

    // check if this is an appropriate hardening.
    if (hardening_) {
      if (!hardening_->IsRateDependent()) {
        mimi::utils::PrintAndThrowError(hardening_->Name(),
                                        "is not rate-dependent.");
      }
      if (!hardening_->IsTemperatureDependent()) {
        mimi::utils::PrintAndThrowError(hardening_->Name(),
                                        "is not temperature-dependent.");
      }

      // validate hardening.
      hardening_->Validate();
    } else {
      mimi::utils::PrintAndThrowError("hardening missing for", Name());
    }

    // temporary solution to getting melt t.
    // TODO do better
    melting_temperature_ =
        std::dynamic_pointer_cast<JohnsonCookAdiabaticRateDependentHardening>(
            hardening_)
            ->melting_temperature_;
  }

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    tmp.aux_mat_.resize(4, mfem::DenseMatrix(dim_, dim_));
  }

  virtual MaterialStatePtr_ CreateState() const {
    MIMI_FUNC();

    std::shared_ptr<State> state = std::make_shared<State>();
    // create 2 matrices with the size of dim x dim and zero initialize
    state->matrices.resize(state->k_state_matrices);
    for (mfem::DenseMatrix& mat : state->matrices) {
      mat.SetSize(dim_, dim_);
      mat = 0.;
    }
    // one scalar, also zero
    state->scalars_.resize(state->k_state_scalars, 0.);

    // set initial temp
    state->scalars_[State::k_temperature] = initial_temperature_;
    return state;
  };

  /// hard coded version of plastic strain rate and temperature computation
  void PlasticStrainRateAndTemperatureRate(const mfem::DenseMatrix& F_dot,
                                           const mfem::DenseMatrix& eps,
                                           double& plastic_strain_rate,
                                           double& temperature_rate) const {
    MIMI_FUNC()
    const double* f = F_dot.GetData();
    const double* e = eps.GetData();

    if (dim_ == 2) {
      // sym(F_dot)
      // for large strain, use L
      const double f0 = f[0];
      const double f3 = f[3];
      const double trace_over_dim = (f0 + f3) / 2.0;

      const double ed0 = f0 - trace_over_dim;
      const double ed1 = .5 * (f[1] + f[2]);
      const double ed3 = f3 - trace_over_dim;

      // get plastic strain rate
      plastic_strain_rate =
          std::sqrt(2. / 3. * (ed0 * ed0 + 2.0 * ed1 * ed1 + ed3 * ed3));

      // Now, temperature
      // Literatures use stress : eps_dot
      // But it doesn't converge and they are negative, which isn't allowed
      //
      // first, stress using lambda * tr(eps) * I + 2*mu*eps
      const double diag = lambda_ * eps.Trace();
      const double two_mu = 2. * mu_;

      //      double sig_eq0 = diag + two_mu * e[0];
      //      double sig_eq3 = diag + two_mu * e[3];
      //      const double trace_over_dim = (sig_eq0 + sig_eq3) / 2.0;
      //      sig_eq0 -= trace_over_dim;
      //      sig_eq3 -= trace_over_dim;
      //      const double sig_eq1 = two_mu * e[1];
      //      const double sig_eq2 = two_mu * e[2];
      //      const double equivalent_stress =
      //          std::sqrt(3. / 2.
      //                    * ((sig_eq0 * sig_eq0) + (sig_eq1 * sig_eq1)
      //                       + (sig_eq2 * sig_eq2) + (sig_eq3 * sig_eq3)));
      //      temperature_rate = heat_fraction_ * equivalent_stress
      //                         * plastic_strain_rate / (density_ *
      //                         specific_heat_);

      // here's alternative approach using abaqus
      const double e0 = e[0];
      const double e3 = e[3];
      // const double t_over_dim = (e0 + e3) * 0.5;
      double sig0 = diag + two_mu * (e0); // - t_over_dim);
      double sig3 = diag + two_mu * (e3); // - t_over_dim);
      const double sig1 = two_mu * e[1];
      const double sig2 = two_mu * e[2];
      const double sig_trace_over_dim = (sig0 + sig3) / 2.0;
      sig0 -= sig_trace_over_dim;
      sig3 -= sig_trace_over_dim;

      const double work = (ed0 * sig0 + ed1 * sig1 + ed1 * sig2 + ed3 * sig3);
      if (work < 0.0) {
        temperature_rate = 0.0;
      } else {
        temperature_rate = heat_fraction_ * work / (density_ * specific_heat_);
      }

      return;
    } else {
      // get eps_dot
      const double ed0 = f[0];

      const double ed1 = 0.5 * (f[1] + f[3]);
      const double ed2 = 0.5 * (f[2] + f[6]);

      const double ed4 = f[4];

      const double ed5 = 0.5 * (f[5] + f[7]);

      const double ed8 = f[8];

      plastic_strain_rate =
          std::sqrt(2. / 3.
                    * (ed0 * ed0 + 2. * ed1 * ed1 + 2. * ed2 * ed2 + ed4 * ed4
                       + 2. * ed5 * ed5 + ed8 * ed8));

      // get stress
      const double diag = lambda_ * eps.Trace();
      const double two_mu = 2. * mu_;
      const double sig_eps_dot =
          ((diag + two_mu * e[0]) * ed0) + ((two_mu * e[1]) * ed1)
          + ((two_mu * e[2]) * ed2) + ((two_mu * e[3]) * ed1)
          + ((diag + two_mu * e[4]) * ed4) + ((two_mu * e[5]) * ed5)
          + ((two_mu * e[6]) * ed2) + ((two_mu * e[7]) * ed5)
          + ((diag + two_mu * e[8]) * ed8);

      temperature_rate =
          (heat_fraction_ * sig_eps_dot) / (density_ * specific_heat_);
    }
  }

  template<bool accumulate>
  void PlasticStress(std::conditional_t<accumulate,
                                        MaterialStatePtr_,
                                        const MaterialStatePtr_>& state,
                     WorkData_& tmp,
                     mfem::DenseMatrix& sigma) const {
    MIMI_FUNC()

    // get aux
    mfem::DenseMatrix& eps = tmp.aux_mat_[k_eps];
    mfem::DenseMatrix& s = tmp.aux_mat_[k_s];
    mfem::DenseMatrix& N_p = tmp.aux_mat_[k_N_p];
    mfem::DenseMatrix& mat_w0 = tmp.aux_mat_[k_work0];

    // get states
    auto& plastic_strain = state->matrices[State::k_plastic_strain];
    auto& accumulated_plastic_strain =
        state->scalars_[State::k_accumulated_plastic_strain];
    auto& temperature = state->scalars_[State::k_temperature];
    auto& previous_eps = state->matrices[State::k_elastic_strain];

    // precompute aux values
    // eps, p, s, eta, q, equivalent plastic strain rate
    ElasticStrain(tmp.F_, plastic_strain, eps);
    const double p = K_ * eps.Trace();
    Dev(eps, dim_, 2.0 * G_, s);
    const double q = sqrt_3_2_ * Norm(s); // trial mises

    const double* s_d = s.GetData();
    const double* eps_d = eps.GetData();
    const double* peps_d = previous_eps.GetData();
    const double* ps_d = plastic_strain.GetData();
    const double fac0 = 1.5 / q * accumulated_plastic_strain;
    const double t_fac1 = 1. / dt_;
    double rate_accum{}, rate_i;
    for (int i{}; i < dim_ * dim_; ++i) {
      rate_i = (fac0 * s_d[i] + eps_d[i] - peps_d[i] - ps_d[i]) * t_fac1;
      rate_accum += rate_i * rate_i;
    }
    const double eqps_rate = std::sqrt(2. / 3. * rate_accum);

    // admissibility
    const double eqps_old = accumulated_plastic_strain;
    const double thermo_visco_contrib =
        hardening_->ThermoContribution(temperature)
        * hardening_->ViscoContribution(eqps_rate);
    auto residual = [eqps_old, q, thermo_visco_contrib, *this](
                        auto delta_eqps) -> ADScalar_ {
      return q - 3.0 * G_ * delta_eqps
             - hardening_->Evaluate(eqps_old + delta_eqps)
                   * thermo_visco_contrib;
    };

    const double tolerance = hardening_->SigmaY() * k_tol;

    if (residual(0.0) > tolerance) {
      /// return mapping
      mimi::solvers::ScalarSolverOptions opts{.xtol = 0.,
                                              .rtol = tolerance,
                                              .max_iter = 100};

      const double lower_bound = 0.0;
      const double upper_bound =
          (q
           - hardening_->Evaluate(eqps_old, eqps_rate, temperature).GetValue())
          / (3.0 * G_);
      const double delta_eqps = mimi::solvers::ScalarSolve(residual,
                                                           0.0,
                                                           lower_bound,
                                                           upper_bound,
                                                           opts);
      N_p.Set(1.5 / q, s); // flow dir
      if constexpr (!accumulate) {
        s.Add(-2.0 * G_ * delta_eqps, N_p);
      }
      if constexpr (accumulate) {
        accumulated_plastic_strain += delta_eqps;
        plastic_strain.Add(delta_eqps, N_p);
        mat_w0.Set(delta_eqps / dt_,
                   N_p); // accumulation rate to get temperature
      }
    }

    // returning s + p * I
    mfem::Add(s,
              tmp.I_,
              p,
              sigma); // we'll compute temperature increase based on this
    if constexpr (accumulate) {
      // in case of accumulation, we save elastic strain as well
      previous_eps = eps;
      const double temp_rate = (heat_fraction_ * std::max(sigma * mat_w0, 0.0))
                               / (density_ * specific_heat_);
      temperature =
          std::min(temperature + temp_rate * dt_, melting_temperature_ + 1.0);
    }
  }

  virtual void EvaluateCauchy(const MaterialStatePtr_& state,
                              WorkData_& tmp,
                              mfem::DenseMatrix& sigma) const {
    MIMI_FUNC()

    PlasticStress<false>(state, tmp, sigma);
  }

  virtual void Accumulate(MaterialStatePtr_& state, WorkData_& tmp) const {
    MIMI_FUNC()

    PlasticStress<true>(state, tmp, tmp.stress_ /* placeholder */);
  }
};

class J2AdiabaticViscoLarge : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;
  using HardeningPtr_ = std::shared_ptr<HardeningBase>;
  using ADScalar_ = typename HardeningBase::ADScalar_;

  // additional parameters
  HardeningPtr_ hardening_;

  // thermo related
  double heat_fraction_{0.9};        // 0.9 is apparently abaqus' default
  double specific_heat_;             // for now, constant
  double initial_temperature_{20.0}; //
  double melting_temperature_{};     // takes from hardening - TODO - change

  static constexpr const double k_tol{1.e-10};

  struct State : public MaterialState {
    static constexpr const int k_state_matrices{2};
    static constexpr const int k_state_scalars{2};
    /// matrix indices
    static constexpr const int k_be_old{0};
    static constexpr const int k_F_old{1};
    /// scalar indices
    static constexpr const int k_accumulated_plastic_strain{0};
    static constexpr const int k_temperature{1};
  };

protected:
  /// some constants
  const double sqrt_3_2_ = std::sqrt(3.0 / 2.0);

  /// lookup index for matrix
  static constexpr const int k_n_aux_mat{3};

public:
  virtual std::string Name() const { return "J2AdiabaticViscoLarge"; }

  /// answers if this material is suitable for visco-solids.
  virtual bool IsRateDependent() const { return true; }

  virtual void Setup(const int dim) {
    MIMI_FUNC()

    /// base setup for conversions and dim, nthread
    Base_::Setup(dim);

    // check if this is an appropriate hardening.
    if (hardening_) {
      if (!hardening_->IsRateDependent()) {
        mimi::utils::PrintAndThrowError(hardening_->Name(),
                                        "is not rate-dependent.");
      }
      if (!hardening_->IsTemperatureDependent()) {
        mimi::utils::PrintAndThrowError(hardening_->Name(),
                                        "is not temperature-dependent.");
      }

      // validate hardening.
      hardening_->Validate();
    } else {
      mimi::utils::PrintAndThrowError("hardening missing for", Name());
    }

    // temporary solution to getting melt t.
    // TODO do better
    melting_temperature_ =
        std::dynamic_pointer_cast<JohnsonCookAdiabaticRateDependentHardening>(
            hardening_)
            ->melting_temperature_;
  }

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    tmp.aux_mat_.assign(k_n_aux_mat, mfem::DenseMatrix(dim_, dim_));
  }

  virtual MaterialStatePtr_ CreateState() const {
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
    state->scalars_.assign(state->k_state_scalars, 0.);

    // set initial temp
    state->scalars_[State::k_temperature] = initial_temperature_;
    return state;
  };

  template<bool accumulate>
  void PlasticStress(std::conditional_t<accumulate,
                                        MaterialStatePtr_,
                                        const MaterialStatePtr_>& state,
                     WorkData_& tmp,
                     mfem::DenseMatrix& P) const {
    MIMI_FUNC()
    using Mat = mfem::DenseMatrix;

    // get aux
    Mat& w_mat0 = tmp.aux_mat_[0];
    Mat& w_mat1 = tmp.aux_mat_[1];
    Mat& w_mat2 = tmp.aux_mat_[2];

    // get states
    // scalars
    auto& accumulated_plastic_strain =
        state->scalars_[State::k_accumulated_plastic_strain];
    auto& temperature = state->scalars_[State::k_temperature];
    // matrix
    auto& F_old = state->matrices[State::k_F_old];
    auto& be_old = state->matrices[State::k_be_old];

    // // now, f = F F_hat^-1
    Mat& f_inv = w_mat0;
    mfem::Mult(F_old, tmp.FInv(), f_inv);

    // f_bar = f / (f.det)^(1/3)
    // f = F F_hat^-1
    Mat& f_bar = w_mat1;
    mfem::CalcInverse(f_inv, f_bar);
    f_bar *= std::cbrt(f_bar.Det());

    // elastic predictor
    Mat& be = w_mat0; // this one stays
    Mat& fbbo = w_mat2;
    // be = f_bar @ be_old @ f_bar^T
    mfem::Mult(f_bar, be_old, fbbo);
    mfem::MultABt(fbbo, f_bar, be);

    // compute strain rate
    Mat& strain_rate = w_mat2;
    mfem::Add(be, be_old, -1., strain_rate);
    strain_rate *= 1. / dt_;
    const double eqps_rate = EquivalentPlasticStrainRate(strain_rate);

    Mat& s = w_mat1;   // this one stays
    Mat& N_p = w_mat2; // this one stays for awhile
    Dev(be, dim_, G_, s);
    const double s_norm = Norm(s);
    if (AlmostZero(s_norm)) {
      const double sqrt_1_2 = std::sqrt(1. / 2.);
      N_p.Diag(sqrt_1_2, dim_);
    } else {
      N_p.Set(std::sqrt(3. / 2.) / s_norm, s);
    }
    const double s_effective = N_p * s; // q

    // admissibility
    const double eqps_old = accumulated_plastic_strain;
    const double thermo_visco_contrib =
        hardening_->ThermoContribution(temperature)
        * hardening_->ViscoContribution(eqps_rate);
    const double be_trace = be.Trace();
    auto residual = [eqps_old,
                     G = G_,
                     s_effective,
                     thermo_visco_contrib,
                     hardening = hardening_,
                     be_trace](auto delta_eqps) -> ADScalar_ {
      // no 3 * G -> (1/3) integrated to be_trace
      return s_effective - G * delta_eqps * be_trace
             - hardening->Evaluate(eqps_old + delta_eqps)
                   * thermo_visco_contrib;
    };

    const double tolerance = hardening_->SigmaY() * k_tol;
    if (residual(0.0) > tolerance) {
      /// return mapping
      mimi::solvers::ScalarSolverOptions opts{.xtol = 0.,
                                              .rtol = tolerance,
                                              .max_iter = 100};

      const double lower_bound = 0.0;
      const double upper_bound =
          (s_effective
           - hardening_->Evaluate(eqps_old).GetValue() * thermo_visco_contrib)
          / (G_ * be_trace);
      const double delta_eqps = mimi::solvers::ScalarSolve(residual,
                                                           0.0,
                                                           lower_bound,
                                                           upper_bound,
                                                           opts);
      be.Add(-2. / 3. * delta_eqps * be_trace, N_p);
      Dev(be, dim_, G_, s);
      if constexpr (accumulate) {
        accumulated_plastic_strain += delta_eqps;
      }
    }

    if constexpr (!accumulate) {
      const double det_F = tmp.DetF();
      // get kirchhoff stress
      Mat& tau = w_mat2;
      mfem::Add(s, tmp.I_, K_ * (det_F * det_F - 1.) * .5, tau);
      mfem::MultABt(tau, tmp.FInv(), P);
    } else { /* accumulate */
      F_old = tmp.F_;
      be_old = be;
      // for current use case, time step is so small that temperature stays
      // almost constant
      // // let's use tau and
      // const double temp_rate = (heat_fraction_ * std::max(tau * be, 0.0))
      //                          / (density_ * specific_heat_);
      // temperature =
      //     std::min(temperature + temp_rate * dt_, melting_temperature_ + 1.);
    }
  }

  virtual void EvaluatePK1(const MaterialStatePtr_& state,
                           WorkData_& tmp,
                           mfem::DenseMatrix& P) const {
    MIMI_FUNC()

    PlasticStress<false>(state, tmp, P);
  }

  virtual void Accumulate(MaterialStatePtr_& state, WorkData_& tmp) const {
    MIMI_FUNC()

    PlasticStress<true>(state, tmp, tmp.stress_ /* placeholder */);
  }
};

/// @brief Implementations for finite strain based on implementations in
/// https://github.com/sandialabs/optimism
/// most of the setup is identical to adiabatic visco
class J2AdiabaticViscoLogStrain : public J2AdiabaticVisco {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;
  using HardeningPtr_ = std::shared_ptr<HardeningBase>;
  using ADScalar_ = typename HardeningBase::ADScalar_;

  struct State : public MaterialState {
    static constexpr const int k_state_matrices{1};
    static constexpr const int k_state_scalars{3};
    /// matrix indices
    static constexpr const int k_plastic_strain{0};
    /// scalar indices
    static constexpr const int k_accumulated_plastic_strain{0};
    static constexpr const int k_temperature{1};
    static constexpr const int k_eqps{
        2}; // try previous step's eqps to compute eqps_dot
  };

  virtual std::string Name() const { return "J2AdiabaticViscoLogStrain"; }

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    // eps, s, N_p, a misc work array, eigen vectors/work2
    tmp.aux_mat_.resize(5, mfem::DenseMatrix(dim_, dim_));
    // eigen values
    tmp.aux_vec_.resize(1, mfem::Vector(dim_));
  }

  virtual MaterialStatePtr_ CreateState() const {
    MIMI_FUNC()

    std::shared_ptr<State> state = std::make_shared<State>();
    // create 2 matrices with the size of dim x dim and zero initialize
    state->matrices.resize(state->k_state_matrices);
    for (mfem::DenseMatrix& mat : state->matrices) {
      mat.SetSize(dim_, dim_);
      mat = 0.;
    }

    // for logarithmic this is I
    state->matrices[State::k_plastic_strain].Diag(1., dim_);

    // one scalar, also zero
    state->scalars_.resize(state->k_state_scalars, 0.);

    // set initial temp
    state->scalars_[State::k_temperature] = initial_temperature_;
    return state;
  }

  template<bool accumulate>
  void PlasticStress(std::conditional_t<accumulate,
                                        MaterialStatePtr_,
                                        const MaterialStatePtr_>& state,
                     WorkData_& tmp,
                     mfem::DenseMatrix& sigma) const {
    MIMI_FUNC()

    // get aux
    mfem::DenseMatrix& eps = tmp.aux_mat_[k_eps];
    mfem::DenseMatrix& s = tmp.aux_mat_[k_s];
    mfem::DenseMatrix& N_p = tmp.aux_mat_[k_N_p];

    // get states
    auto& plastic_strain = state->matrices[State::k_plastic_strain];
    auto& accumulated_plastic_strain =
        state->scalars_[State::k_accumulated_plastic_strain];
    auto& temperature = state->scalars_[State::k_temperature];
    auto& eqps = state->scalars_[State::k_eqps];

    // precompute aux values
    LogarithmicStrain(plastic_strain, tmp, eps);
    const double p = K_ * eps.Trace();
    Dev(eps, dim_, 2.0 * G_, s);
    const double q = sqrt_3_2_ * Norm(s); // trial mises
    // get eqps_rate and delta temperature
    double eqps_rate, temperature_rate;
    PlasticStrainRateAndTemperatureRate(tmp.F_dot_,
                                        eps,
                                        eqps_rate,
                                        temperature_rate);

    // admissibility
    const double eqps_old = accumulated_plastic_strain;
    // const double eqps_rate = EquivalentPlasticStrainRate(tmp.F_dot_);
    const double trial_T =
        temperature + temperature_rate * second_effective_dt_;
    auto residual =
        [eqps_old, eqps_rate, q, trial_T, *this](auto delta_eqps) -> ADScalar_ {
      return q - 3.0 * G_ * delta_eqps
             - hardening_->Evaluate(eqps_old + delta_eqps, eqps_rate, trial_T);
    };
    const double tolerance = hardening_->SigmaY() * k_tol;

    if (residual(0.0) > tolerance) {
      /// return mapping
      mimi::solvers::ScalarSolverOptions opts{.xtol = 0.,
                                              .rtol = tolerance,
                                              .max_iter = 100};

      const double lower_bound = 0.0;
      const double upper_bound =
          (q
           - hardening_->Evaluate(eqps_old, eqps_rate, temperature).GetValue())
          / (3.0 * G_);
      const double delta_eqps = mimi::solvers::ScalarSolve(residual,
                                                           0.0,
                                                           lower_bound,
                                                           upper_bound,
                                                           opts);
      N_p.Set(1.5 / q, s);
      if constexpr (!accumulate) {
        s.Add(-2.0 * G_ * delta_eqps, N_p);
      }
      if constexpr (accumulate) {
        // plastic_strain.Add(delta_eqps, N_p);
        // for logarithmic, we do the following instead
        accumulated_plastic_strain += delta_eqps;

        constexpr const int k_work0{3};
        constexpr const int k_work1{4};
        constexpr const int k_eig_vec{0};

        mfem::DenseMatrix& work0 = tmp.aux_mat_[k_work0];
        mfem::DenseMatrix& work1 = tmp.aux_mat_[k_work1];

        mfem::DenseMatrix& increment = work0;
        increment.Set(delta_eqps, N_p);
        increment.Symmetrize();

        mfem::Vector& eigen_values = tmp.aux_vec_[k_eig_vec];
        mfem::DenseMatrix& eigen_vectors = work1;
        increment.CalcEigenvalues(eigen_values.GetData(),
                                  eigen_vectors.GetData());
        // apply exp
        for (int i{}; i < dim_; ++i) {
          eigen_values[i] = std::exp(eigen_values[i]);
        }
        mfem::DenseMatrix& exp_symm = work0; // reuse
        mfem::MultADAt(eigen_vectors, eigen_values, exp_symm);

        mfem::DenseMatrix& old_plastic_strain = work1; // reuse
        old_plastic_strain = plastic_strain;
        mfem::Mult(exp_symm, old_plastic_strain, plastic_strain);

        // now, temp
        const double temp_rate = (heat_fraction_ * q * (delta_eqps / dt_)
                                  / density_ * specific_heat_);
        temperature = std::min(trial_T, melting_temperature_ + 1.0);
        // mimi::utils::PrintSynced(temp_rate,
        //                          q,
        //                          delta_eqps,
        //                          delta_eqps / dt_,
        //                          temperature);
      }
    }

    // returning s + p * I
    if constexpr (!accumulate) {
      mfem::Add(s, tmp.I_, p / tmp.DetF(), sigma);
      // mfem::Add(s, tmp.I_, p, sigma);
      // sigma *= 1. / tmp.DetF();
    }
  }

  virtual void EvaluateCauchy(const MaterialStatePtr_& state,
                              WorkData_& tmp,
                              mfem::DenseMatrix& sigma) const {
    MIMI_FUNC()

    PlasticStress<false>(state, tmp, sigma);
  }

  virtual void Accumulate(MaterialStatePtr_& state, WorkData_& tmp) const {
    MIMI_FUNC()

    PlasticStress<true>(state, tmp, tmp.stresslowers _ /* placeholder */);
  }
};

} // namespace mimi::materials
