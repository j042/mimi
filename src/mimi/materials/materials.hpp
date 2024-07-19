#pragma once

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

  /// defining material properties using young's modulus and poisson's ratio
  virtual void SetYoungPoisson(const double young, const double poisson);

  /// defining the material properties using lame parameters
  virtual void SetLame(const double lambda, const double mu);

  /// self setup. will be called once.
  /// unless you want to do it your self, call this one.
  /// before you extend Setup.
  virtual void Setup(const int dim);

  virtual void AllocateAux(WorkData_& tmp) const {
    mimi::utils::PrintAndThrowError("AllocateAux not implemented for", Name());
  }

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
                              mfem::DenseMatrix& sigma) const;

  /// @brief unless implemented, this will try to call evaluate sigma and
  /// transform if none of stress is implemented, you will be stuck in a
  /// neverending loop current implementation is not so memory efficient
  virtual void EvaluatePK1(const MaterialStatePtr_& state,
                           WorkData_& tmp,
                           mfem::DenseMatrix& P) const;

  /// state accumulating version
  virtual void Accumulate(MaterialStatePtr_& state, WorkData_& tmp) const {
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
                           mfem::DenseMatrix& P) const;
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
                              mfem::DenseMatrix& sigma) const;
};

/// @brief Computational Methods for plasticity p260, box 7.5
/// Implementation referenced from serac
class J2Linear : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;

  // additional parameters
  double isotropic_hardening_;
  double kinematic_hardening_;
  double sigma_y_; // yield

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

  virtual MaterialStatePtr_ CreateState() const;

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
        state->scalars[State::k_accumulated_plastic_strain];

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

  // thermo related
  double heat_fraction_{0.9};        // 0.9 is apparently abaqus' default
  double specific_heat_;             // for now, constant
  double initial_temperature_{20.0}; //
  double melting_temperature_{-1.0};

  static constexpr const double k_tol{1.e-10};

  struct State : public MaterialState {
    static constexpr const int k_state_matrices{1};
    static constexpr const int k_state_scalars{2};
    /// matrix indices
    static constexpr const int k_plastic_strain{0};
    /// scalar indices
    static constexpr const int k_accumulated_plastic_strain{0};
    static constexpr const int k_temperature{1};
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
  virtual std::string Name() const { return "J2"; }

  virtual void Setup(const int dim);

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    tmp.aux_mat_.resize(3, mfem::DenseMatrix(dim_, dim_));
  }

  virtual MaterialStatePtr_ CreateState() const;

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
        state->scalars[State::k_accumulated_plastic_strain];

    // precompute aux values
    // eps, p, s, eta, q, phi
    ElasticStrain(tmp.F_, plastic_strain, eps);
    const double p = K_ * eps.Trace();
    Dev(eps, dim_, 2.0 * G_, s);
    const double q = sqrt_3_2_ * Norm(s);

    // get values required for return mapping
    const double eqps_old = accumulated_plastic_strain;
    // this may be just one
    const double thermo_contribution =
        hardening_->ThermoContribution(state->scalars[State::k_temperature]);

    // admissibility
    auto residual =
        [eqps_old, q, this, thermo_contribution](auto delta_eqps) -> ADScalar_ {
      return q - 3.0 * G_ * delta_eqps
             - hardening_->Evaluate(eqps_old + delta_eqps)
                   * (hardening_->RateContribution(delta_eqps / dt_)
                      * thermo_contribution);
    };

    const double tolerance = hardening_->SigmaY() * k_tol;

    if (residual(0.0) > tolerance) {
      /// return mapping
      mimi::solvers::ScalarSolverOptions opts{.xtol = 0.,
                                              .rtol = tolerance,
                                              .max_iter = 100};

      const double lower_bound = 0.0;
      const double upper_bound =
          (q - hardening_->Evaluate(eqps_old).GetValue() * thermo_contribution)
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

      // no accumulation here
      if constexpr (accumulate) {
        accumulated_plastic_strain += delta_eqps;
        plastic_strain.Add(delta_eqps, N_p);

        // update temp if needed
        if (hardening_->IsTemperatureDependent()) {
          state->scalars[State::k_temperature] +=
              heat_fraction_ * q * delta_eqps / (density_ * specific_heat_);
        }
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

class J2Simo : public MaterialBase {
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
  double melting_temperature_{-1.0};

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
  virtual std::string Name() const { return "J2Simo"; }

  virtual void Setup(const int dim);

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    tmp.aux_mat_.assign(k_n_aux_mat, mfem::DenseMatrix(dim_, dim_));
  }

  virtual MaterialStatePtr_ CreateState() const;

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
        state->scalars[State::k_accumulated_plastic_strain];
    auto& temperature = state->scalars[State::k_temperature];
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
    const double thermo_contrib = hardening_->ThermoContribution(temperature);
    const double be_trace = be.Trace();
    auto residual = [eqps_old,
                     G = G_,
                     s_effective,
                     thermo_contrib,
                     hardening = hardening_,
                     be_trace,
                     this](auto delta_eqps) -> ADScalar_ {
      // no 3 * G -> (1/3) integrated to be_trace
      return s_effective - G * delta_eqps * be_trace
             - hardening->Evaluate(eqps_old + delta_eqps)
                   * (thermo_contrib
                      * hardening_->RateContribution(delta_eqps / dt_));
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
           - hardening_->Evaluate(eqps_old).GetValue() * thermo_contrib)
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
        if (hardening_->IsTemperatureDependent()) {
          state->scalars[State::k_temperature] += heat_fraction_ * s_effective
                                                  * delta_eqps
                                                  / (density_ * specific_heat_);
        }
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
/// github.com/sandialabs/optimism & github.com/LLNL/serac
class J2Log : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;
  using HardeningPtr_ = std::shared_ptr<HardeningBase>;
  using ADScalar_ = typename HardeningBase::ADScalar_;

  // aux mat consts
  static constexpr const int k_E_e{0};
  static constexpr const int k_s{1};
  static constexpr const int k_N_p{2};
  static constexpr const int k_F_e{3};
  static constexpr const int k_work0{4};
  static constexpr const int k_work1{5};
  // aux vec consts
  static constexpr const int k_eig_vec{0};

  static constexpr const double k_tol{1.e-10};

  // additional parameters
  HardeningPtr_ hardening_;

  // thermo related
  double heat_fraction_{0.9};        // 0.9 is apparently abaqus' default
  double specific_heat_;             // for now, constant
  double initial_temperature_{20.0}; //
  double melting_temperature_{-1.0};

  struct State : public MaterialState {
    static constexpr const int k_state_matrices{1};
    static constexpr const int k_state_scalars{2};
    /// matrix indices
    static constexpr const int k_Fp_inv{0};
    /// scalar indices
    static constexpr const int k_accumulated_plastic_strain{0};
    static constexpr const int k_temperature{1};
  };

  virtual std::string Name() const { return "J2Log"; }

  virtual void Setup(const int dim);

  virtual void AllocateAux(WorkData_& tmp) const {
    MIMI_FUNC()

    // Ee, s, N_p, a misc work array, eigen vectors/work2
    tmp.aux_mat_.resize(5, mfem::DenseMatrix(dim_, dim_));
    // eigen values
    tmp.aux_vec_.resize(1, mfem::Vector(dim_));
  }

  virtual MaterialStatePtr_ CreateState() const;

  template<bool accumulate>
  void PlasticStress(std::conditional_t<accumulate,
                                        MaterialStatePtr_,
                                        const MaterialStatePtr_>& state,
                     WorkData_& tmp,
                     mfem::DenseMatrix& sigma) const {
    MIMI_FUNC()

    // get aux
    mfem::DenseMatrix& E_e = tmp.aux_mat_[k_E_e];
    mfem::DenseMatrix& s = tmp.aux_mat_[k_s];
    mfem::DenseMatrix& N_p = tmp.aux_mat_[k_N_p];
    mfem::DenseMatrix& F_e = tmp.aux_mat_[k_F_e];

    // get states
    auto& Fp_inv = state->matrices[State::k_Fp_inv];
    auto& accumulated_plastic_strain =
        state->scalars[State::k_accumulated_plastic_strain];
    auto& temperature = state->scalars[State::k_temperature];

    mfem::Mult(tmp.F_, Fp_inv, F_e);
    LogarithmicStrain<k_work0, k_work1, k_eig_vec>(F_e, tmp, E_e);
    const double p = K_ * E_e.Trace();
    Dev(E_e, dim_, 2.0 * G_, s);
    const double q = sqrt_3_2_ * Norm(s); // trial mises

    // admissibility
    const double eqps_old = accumulated_plastic_strain;
    const double thermo_contrib = hardening_->ThermoContribution(temperature);
    auto residual =
        [eqps_old, q, this, thermo_contrib](auto delta_eqps) -> ADScalar_ {
      return q - 3.0 * G_ * delta_eqps
             - hardening_->Evaluate(eqps_old + delta_eqps)
                   * (hardening_->RateContribution(delta_eqps / dt_)
                      * thermo_contrib);
    };
    const double tolerance = hardening_->SigmaY() * k_tol;

    mfem::DenseMatrix& work0 = tmp.aux_mat_[k_work0];
    mfem::DenseMatrix& work1 = tmp.aux_mat_[k_work1];

    if (residual(0.0) > tolerance) {
      /// return mapping
      mimi::solvers::ScalarSolverOptions opts{.xtol = 0.,
                                              .rtol = tolerance,
                                              .max_iter = 100};

      const double lower_bound = 0.0;
      const double upper_bound =
          (q - hardening_->Evaluate(eqps_old).GetValue() * thermo_contrib)
          / (3.0 * G_);
      const double delta_eqps = mimi::solvers::ScalarSolve(residual,
                                                           0.0,
                                                           lower_bound,
                                                           upper_bound,
                                                           opts);
      N_p.Set(1.5 / q, s);
      s.Add(-2.0 * G_ * delta_eqps, N_p);
      // for logarithmic, we do the following instead

      mfem::DenseMatrix& increment = work0;
      increment.Set(-delta_eqps, N_p); // sym

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

      // update F_e
      mfem::DenseMatrix& F_e_old = work1;
      F_e_old = F_e;
      mfem::Mult(F_e_old, exp_symm, F_e);

      if constexpr (accumulate) {
        accumulated_plastic_strain += delta_eqps;

        mfem::DenseMatrix& Fp_inv_old = work1;
        Fp_inv_old = Fp_inv;
        mfem::Mult(Fp_inv_old, exp_symm, Fp_inv);
      }
    }

    if constexpr (!accumulate) {
      // M = s + p * I
      mfem::Add(s, tmp.I_, p / tmp.DetF(), tmp.alternative_stress_);
      // sigma = F_e^-T * M * F_e^T / det(F)
      // P = det(F) * sigma * F^-T
      // --> P = F_e^-T M F_e^T * F^-T
      mfem::CalcInverse(F_e, work0);
      mfem::MultAtB(work0, tmp.alternative_stress_, work1);
      mfem::MultABt(work1, F_e, work0);
      mfem::MultABt(work0, tmp.FInv(), tmp.stress_);
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

private:
  const double sqrt_3_2_ = std::sqrt(1.5);
};

} // namespace mimi::materials
