#pragma once

#include <algorithm>
#include <cmath>

#include <mfem.hpp>

#include "mimi/solvers/newton.hpp"
#include "mimi/utils/ad.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

template<typename T>
using Vector_ = mimi::utils::Vector<T>;

/// saves material state at quadrature point
struct MaterialState {
  Vector_<mfem::DenseMatrix> matrices_;
  Vector_<mfem::Vector> vectors_;
  Vector_<double> scalars_;

  /// flag to notify non-accumulation
  /// use during in FD and line search
  bool freeze_{false};
};

/// Material base.
/// We can either define material or material state (in this case one material)
/// at each quad point. Something to consider before implementing everything
class MaterialBase {
public:
  using MaterialStatePtr_ = std::shared_ptr<MaterialState>;

protected:
  int dim_;
  int n_threads_;

  /// those tmp/aux containers needs to be initialized in Setup
  /// for thread safe use.
  Vector_<Vector_<mfem::DenseMatrix>> aux_matrices_;
  Vector_<Vector_<mfem::Vector>> aux_vectors_;

  /// this is to help switch from PK1 to sigma and vice versa
  Vector_<Vector_<mfem::DenseMatrix>> stress_conversions_;

  static constexpr const int k_P{0};
  static constexpr const int k_sigma{1};
  static constexpr const int k_F_inv{2};

public:
  double density_{-1.0};
  double viscosity_{-1.0};
  double lambda_{-1.0};
  double mu_{-1.0};

  virtual void Prepare(const int dim, const int n_threads_) { MIMI_FUNC() }

  virtual std::string Name() const { return "MaterialBase"; }

  /// self setup. will be called once.
  /// unless you want to do it your self, call this one.
  /// before you extend Setup.
  virtual void Setup(const int dim, const int n_threads) {
    MIMI_FUNC()

    dim_ = dim;
    n_threads_ = n_threads;

    stress_conversions_.resize(
        n_threads_,
        Vector_<mfem::DenseMatrix>(3, mfem::DenseMatrix(dim_)));
  }

  /// answers if this material is suitable for visco-solids.
  virtual bool IsRateDependent() const { return false; }

  /// gives hint to integrator, which stress is implemented we need
  virtual bool UsesCauchy() const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("Please override UsesCauchy() function");
    return false;
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
  virtual void EvaluateCauchy(const mfem::DenseMatrix& F,
                              const int& i_thread,
                              MaterialStatePtr_& state,
                              mfem::DenseMatrix& sigma) {
    MIMI_FUNC()

    // setup aux
    auto& i_conv = stress_conversions_[i_thread];
    mfem::DenseMatrix& P = i_conv[k_P];

    // get P
    EvaluatePK1(F, i_thread, state, P);

    // 1 / det(F) * P * F^T
    // they don't have mfem::Mult_a_ABt();
    mfem::MultABt(P, F, sigma);
    sigma *= 1. / F.Det();
  }

  virtual void EvaluateCauchy(const mfem::DenseMatrix& F,
                              const mfem::DenseMatrix& F_dot,
                              const int& i_thread,
                              MaterialStatePtr_& state,
                              mfem::DenseMatrix& sigma) {
    MIMI_FUNC()

    // setup aux
    auto& i_conv = stress_conversions_[i_thread];
    mfem::DenseMatrix& P = i_conv[k_P];

    EvaluatePK1(F, F_dot, i_thread, state, sigma);

    // 1 / det(F) * P * F^T
    // they don't have mfem::Mult_a_ABt();
    mfem::MultABt(P, F, sigma);
    sigma *= 1. / F.Det();
  }

  /// @brief unless implemented, this will try to call evaluate sigma and
  /// transform if none of stress is implemented, you will be stuck in a
  /// neverending loop current implementation is not so memory efficient
  virtual void EvaluatePK1(const mfem::DenseMatrix& F,
                           const int& i_thread,
                           MaterialStatePtr_& state,
                           mfem::DenseMatrix& P) {
    MIMI_FUNC()

    // setup aux
    auto& i_conv = stress_conversions_[i_thread];
    mfem::DenseMatrix& F_inv = i_conv[k_F_inv];
    mfem::DenseMatrix& sigma = i_conv[k_sigma];

    // get sigma
    EvaluateCauchy(F, i_thread, state, sigma);

    // P = det(F) * sigma * F^-T
    mfem::CalcInverse(F, F_inv);
    mfem::MultABt(sigma, F_inv, P);
    P *= F.Det();
  }

  virtual void EvaluatePK1(const mfem::DenseMatrix& F,
                           const mfem::DenseMatrix& F_dot,
                           const int& i_thread,
                           MaterialStatePtr_& state,
                           mfem::DenseMatrix& P) {
    MIMI_FUNC()

    // setup aux
    auto& i_conv = stress_conversions_[i_thread];
    mfem::DenseMatrix& F_inv = i_conv[k_F_inv];
    mfem::DenseMatrix& sigma = i_conv[k_sigma];

    // get sigma
    EvaluateCauchy(F, F_dot, i_thread, state, sigma);

    // P = det(F) * sigma * F^-T
    mfem::CalcInverse(F, F_inv);
    mfem::MultABt(sigma, F_inv, P);
    P *= F.Det();
  }

  ///
  virtual void EvaluateGrad() const { MIMI_FUNC() }
};

class StVenantKirchhoff : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;

protected:
  /// I am thread-safe. Don't touch me after Setup
  mfem::DenseMatrix I_;

  /// just for lookup index
  static constexpr const int k_C{0};
  static constexpr const int k_E{1};
  static constexpr const int k_S{2};

public:
  virtual std::string Name() const { return "StVenantKirchhoff"; }

  /// gives hint to integrator, which stress is implemented we need
  virtual bool UsesCauchy() const {
    MIMI_FUNC()
    return false;
  }

  virtual void Setup(const int dim, const int nthread) {
    MIMI_FUNC()

    /// base setup for conversions and dim, nthread
    Base_::Setup(dim, nthread);

    // I
    I_.Diag(1., dim);

    /// make space for C F S
    aux_matrices_.resize(
        n_threads_,
        Vector_<mfem::DenseMatrix>(3, mfem::DenseMatrix(dim_)));
  }

  virtual void EvaluatePK1(const mfem::DenseMatrix& F,
                           const int& i_thread,
                           MaterialStatePtr_&,
                           mfem::DenseMatrix& P) {
    MIMI_FUNC()

    // get aux
    auto& i_aux = aux_matrices_[i_thread];
    mfem::DenseMatrix& C = i_aux[k_C];
    mfem::DenseMatrix& E = i_aux[k_E];
    mfem::DenseMatrix& S = i_aux[k_S];

    // C
    mfem::MultAtB(F, F, C);

    // E
    mfem::Add(.5, C, -.5, I_, E);

    // S
    mfem::Add(lambda_ * E.Trace(), I_, 2 * mu_, E, S);

    // P
    mfem::Mult(F, S, P);
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
  /// I am thread-safe. Don't touch me after Setup
  mfem::DenseMatrix I_;

  /// just for lookup index
  static constexpr const int k_B{0};

public:
  virtual std::string Name() const { return "CompressibleOdgenNeoHookean"; }

  /// gives hint to integrator, which stress is implemented we need
  virtual bool UsesCauchy() const {
    MIMI_FUNC()
    return true;
  }

  virtual void Setup(const int dim, const int nthread) {
    MIMI_FUNC()

    /// base setup for conversions and dim, nthread
    Base_::Setup(dim, nthread);

    // I
    I_.Diag(1., dim);

    /// make space for C F S
    aux_matrices_.resize(
        n_threads_,
        Vector_<mfem::DenseMatrix>(1, mfem::DenseMatrix(dim_)));
  }

  /// mu / det(F) * (B - I) + lambda * (det(F) - 1) I
  virtual void EvaluateCauchy(const mfem::DenseMatrix& F,
                              const int& i_thread,
                              MaterialStatePtr_&,
                              mfem::DenseMatrix& sigma) {
    MIMI_FUNC()

    // get aux
    auto& i_aux = aux_matrices_[i_thread];
    mfem::DenseMatrix& B = i_aux[k_B];

    // precompute aux values
    const double det_F = F.Det();
    const double mu_over_det_F = mu_ / det_F;
    mfem::MultABt(F, F, B); // left green

    // flattening the eq above,
    // mu / det(F) B - mu / det(F) I + lambda * (det(F) - 1) I
    // mu / det(F) B + (-mu / det(F) + lambda * (det(F) - 1)) I
    mfem::Add(mu_over_det_F,
              B,
              -mu_over_det_F + lambda_ * (det_F - 1.),
              I_,
              sigma);
  }
};

/* define some static math helper functions often used in J2 and beyond */

/// computes deviator.
/// often used to compute stress deviator, so there's factor
inline void Dev(const mfem::DenseMatrix& A,
                const int dim,
                const double factor,
                mfem::DenseMatrix& dev_A) {
  // get data ptr
  const double* A_data = A.GetData();
  double* dev_A_data = dev_A.GetData();

  if (dim == 2) {
    const double& A_0 = A_data[0];
    const double& A_3 = A_data[3];
    const double tr_A_over_dim = (A_0 + A_3) / dim;

    dev_A_data[0] = (A_0 - tr_A_over_dim) * factor;
    dev_A_data[1] = A_data[1] * factor;
    dev_A_data[2] = A_data[2] * factor;
    dev_A_data[3] = (A_3 - tr_A_over_dim) * factor;
    return;
  } else {
    const double& A_0 = A_data[0];
    const double& A_4 = A_data[4];
    const double& A_8 = A_data[8];
    const double tr_A_over_dim = (A_0 + A_4 + A_8) / dim;

    dev_A_data[0] = (A_0 - tr_A_over_dim) * factor;
    dev_A_data[1] = A_data[1] * factor;
    dev_A_data[2] = A_data[2] * factor;
    dev_A_data[3] = A_data[3] * factor;
    dev_A_data[4] = (A_4 - tr_A_over_dim) * factor;
    dev_A_data[5] = A_data[5] * factor;
    dev_A_data[6] = A_data[6] * factor;
    dev_A_data[7] = A_data[7] * factor;
    dev_A_data[8] = (A_8 - tr_A_over_dim) * factor;
    return;
  }
}

/// sym(F - I) - plastic strain
/// or , sym(F) - I - plastic strain
inline void ElasticStrain(const mfem::DenseMatrix& F,
                          const mfem::DenseMatrix& plastic_strain,
                          mfem::DenseMatrix& elastic_strain) {
  const int dim = F.Height();
  const double* F_data = F.GetData();
  const double* p_data = plastic_strain.GetData();
  double* e_data = elastic_strain.GetData();

  // first, use space of e to copy F
  std::copy_n(F_data, dim * dim, e_data);

  // symmetrize
  elastic_strain.Symmetrize();

  // substract I
  for (int i{}; i < dim; ++i) {
    e_data[(dim + 1) * i] -= 1;
  }

  // subtract plastic_strain
  for (int i{}; i < dim * dim; ++i) {
    e_data[i] -= p_data[i];
  }
}

/// Frobenius norm of a matrix
inline double Norm(const mfem::DenseMatrix& A) {
  const int dim = A.Height();

  const double* A_data = A.GetData();
  double a{};
  for (int i{}; i < dim * dim; ++i) {
    const double& A_i = A_data[i];
    a += A_i * A_i;
  }
  return std::sqrt(a);
}

/// A[i] -= a;
inline void Subtract(mfem::DenseMatrix& A, const double a) {
  const int dim = A.Height();
  double* A_data = A.GetData();
  for (int i{}; i < dim * dim; ++i) {
    A_data[i] += a;
  }
}

/// A += a * I
inline void AddDiag(mfem::DenseMatrix& A, const double a) {
  const int dim = A.Height();
  double* A_data = A.GetData();
  for (int i{}; i < dim; ++i) {
    A_data[(dim + 1) * i] += a;
  }
}

/// @brief Computational Methods for plasticity p260, box 7.5
/// Implementation reference from serac
class J2 : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;

  // additional parameters
  double isotropic_hardening_;
  double kinematic_hardening_;
  double sigma_y_; // yield

  /// @brief Bulk Modulus (lambda + 2 / 3 mu)
  double K_; // K
  /// shear modulus (=mu)
  double G_;

  /// decided to use inline
  bool is_2d_{true};

  struct State : public MaterialState {
    static constexpr const int k_state_matrices{2};
    static constexpr const int k_state_scalars{2};
    /// matrix indices
    static constexpr const int k_beta{0};
    static constexpr const int k_plastic_strain{1};
    /// scalar indices
    static constexpr const int k_accumulated_plastic_strain{0};
  };

protected:
  /// I am thread-safe. Don't touch me after Setup
  mfem::DenseMatrix I_;

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
  virtual std::string Name() const { return "J2"; }

  /// gives hint to integrator, which stress is implemented we need
  virtual bool UsesCauchy() const {
    MIMI_FUNC()
    return false;
  }

  virtual void Setup(const int dim, const int nthread) {
    MIMI_FUNC()

    /// base setup for conversions and dim, nthread
    Base_::Setup(dim, nthread);

    // I
    I_.Diag(1., dim);

    // match variable name with the literature
    K_ = lambda_ + (2 * mu_ / 3);
    G_ = mu_;

    /// make space for du_dX
    aux_matrices_.resize(
        n_threads_,
        Vector_<mfem::DenseMatrix>(3, mfem::DenseMatrix(dim_)));
  }

  virtual MaterialStatePtr_ CreateState() const {
    MIMI_FUNC();

    std::shared_ptr<State> state = std::make_shared<State>();
    // create 2 matrices with the size of dim x dim and zero initialize
    state->matrices_.resize(state->k_state_matrices);
    for (mfem::DenseMatrix& mat : state->matrices_) {
      mat.SetSize(dim_, dim_);
      mat = 0.;
    }
    // one scalar, also zero
    state->scalars_.resize(state->k_state_scalars, 0.);
    return state;
  };

  virtual void EvaluateCauchy(const mfem::DenseMatrix& F,
                              const int& i_thread,
                              MaterialStatePtr_& state,
                              mfem::DenseMatrix& sigma) {
    MIMI_FUNC()

    // get aux
    auto& i_aux = aux_matrices_[i_thread];
    mfem::DenseMatrix& eps = i_aux[k_eps];
    mfem::DenseMatrix& s = i_aux[k_s];
    mfem::DenseMatrix& eta = i_aux[k_eta];

    // get states
    mfem::DenseMatrix& beta = state->matrices_[State::k_beta];
    mfem::DenseMatrix& plastic_strain =
        state->matrices_[State::k_plastic_strain];
    double& accumulated_plastic_strain =
        state->scalars_[State::k_accumulated_plastic_strain];

    // precompute aux values
    // eps, p, s, eta, q, phi
    ElasticStrain(F, plastic_strain, eps);
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

      // return mapping
      s.Add(-sqrt_6_ * G_ * plastic_strain_inc, eta);

      // this part should only be done at stepping.
      if (!state->freeze_) {
        accumulated_plastic_strain += plastic_strain_inc;
        plastic_strain.Add(sqrt_3_2_ * plastic_strain_inc, eta);
        beta.Add(sqrt_2_3_ * kinematic_hardening_ * plastic_strain_inc, eta);
      }
    }

    // returning s + p * I
    mfem::Add(s, I_, p, sigma);
  }
};

struct HardeningBase {
  using ADScalar_ = mimi::utils::ADScalar<double, 1>;

  std::string Name() const { return "HardeningBase"; }

  bool IsRateDependent() const { return false; }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain,
           const ADScalar_& equivalent_plastic_strain_rate) const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "HardeningBase::Evaluate (rate-dependent) not overriden");
    return {};
  }
  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain) const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("HardeningBase::Evaluate not overriden");
    return {};
  }
  virtual double SigmaY() const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("HardeningBase::SigmaY not overriden");
    return -1.0;
  }
};

struct PowerLawHardening : public HardeningBase {
  using Base_ = HardeningBase;
  using ADScalar_ = Base_::ADScalar_;

  double sigma_y_;
  double n_;
  double eps0_;

  std::string Name() const { return "PowerLawHardening"; }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain) const {
    MIMI_FUNC()

    return sigma_y_ * pow(1.0 + accumulated_plastic_strain / eps0_, 1.0 / n_);
  }

  virtual double SigmaY() const { return sigma_y_; }
};

struct VoceHardening : public HardeningBase {
  using Base_ = HardeningBase;
  using ADScalar_ = Base_::ADScalar_;

  double sigma_y_;
  double sigma_sat_;
  double strain_constant_;

  std::string Name() const { return "VoceHardening"; }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain) const {
    MIMI_FUNC()

    return sigma_sat_
           - (sigma_sat_ - sigma_y_)
                 * exp(-accumulated_plastic_strain / strain_constant_);
  }

  virtual double SigmaY() const { return sigma_y_; }
};

struct JohnsonCookHardening : public HardeningBase {
  using Base_ = HardeningBase;
  using ADScalar_ = Base_::ADScalar_;

  double A_;
  double B_;
  double n_;

  std::string Name() const { return "JohnsonCookHardening"; }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain) const {
    MIMI_FUNC()
    if (std::abs(accumulated_plastic_strain.GetValue()) < 1.e-13) {
      return ADScalar_(A_);
    }

    return A_ + B_ * pow(accumulated_plastic_strain, n_);
  }

  virtual double SigmaY() const { return A_; }
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
class J2NonlinearIsotropicHardening : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;
  using HardeningPtr_ = std::shared_ptr<HardeningBase>;
  using ADScalar_ = typename HardeningBase::ADScalar_;

  // additional parameters
  HardeningPtr_ hardening_;

  /// @brief Bulk Modulus (lambda + 2 / 3 mu)
  double K_; // K
  /// shear modulus (=mu)
  double G_;

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
  /// I am thread-safe. Don't touch me after Setup
  mfem::DenseMatrix I_;

  /// some constants
  const double sqrt_3_2_ = std::sqrt(3.0 / 2.0);

  /// lookup index for matrix
  /// elastic strain
  static constexpr const int k_eps{0};
  /// stress deviator
  static constexpr const int k_s{1};
  /// N_p
  static constexpr const int k_N_p{2};

public:
  virtual std::string Name() const { return "J2NonlinearIsotropicHardening"; }

  /// gives hint to integrator, which stress is implemented we need
  virtual bool UsesCauchy() const {
    MIMI_FUNC()
    return false;
  }

  virtual void Setup(const int dim, const int nthread) {
    MIMI_FUNC()

    /// base setup for conversions and dim, nthread
    Base_::Setup(dim, nthread);

    // I
    I_.Diag(1., dim);

    // match variable name with the literature
    K_ = lambda_ + (2 * mu_ / 3);
    G_ = mu_;

    /// make space for du_dX
    aux_matrices_.resize(
        n_threads_,
        Vector_<mfem::DenseMatrix>(3, mfem::DenseMatrix(dim_)));
  }

  virtual MaterialStatePtr_ CreateState() const {
    MIMI_FUNC();

    std::shared_ptr<State> state = std::make_shared<State>();
    // create 2 matrices with the size of dim x dim and zero initialize
    state->matrices_.resize(state->k_state_matrices);
    for (mfem::DenseMatrix& mat : state->matrices_) {
      mat.SetSize(dim_, dim_);
      mat = 0.;
    }
    // one scalar, also zero
    state->scalars_.resize(state->k_state_scalars, 0.);
    return state;
  };

  virtual void EvaluateCauchy(const mfem::DenseMatrix& F,
                              const int& i_thread,
                              MaterialStatePtr_& state,
                              mfem::DenseMatrix& sigma) {
    MIMI_FUNC()

    // get aux
    auto& i_aux = aux_matrices_[i_thread];
    mfem::DenseMatrix& eps = i_aux[k_eps];
    mfem::DenseMatrix& s = i_aux[k_s];
    mfem::DenseMatrix& N_p = i_aux[k_N_p];

    // get states
    mfem::DenseMatrix& plastic_strain =
        state->matrices_[State::k_plastic_strain];
    double& accumulated_plastic_strain =
        state->scalars_[State::k_accumulated_plastic_strain];

    // precompute aux values
    // eps, p, s, eta, q, phi
    ElasticStrain(F, plastic_strain, eps);
    const double p = K_ * eps.Trace();
    Dev(eps, dim_, 2.0 * G_, s);
    const double q = sqrt_3_2_ * Norm(s);

    // admissibility
    const double eqps_old = accumulated_plastic_strain;
    auto residual = [eqps_old, *this](auto delta_eqps,
                                      auto trial_mises) -> ADScalar_ {
      return trial_mises - 3.0 * G_ * delta_eqps
             - hardening_->Evaluate(eqps_old + delta_eqps);
    };

    const double tolerance = hardening_->SigmaY() * k_tol;

    if (residual(0.0, q) > tolerance) {
      /// return mapping
      mimi::solvers::ScalarSolverOptions opts{.xtol = 0.,
                                              .rtol = tolerance,
                                              .max_iter = 100};

      const double lower_bound = 0.0;
      const double upper_bound =
          (q - hardening_->Evaluate(eqps_old).GetValue() / (3.0 * G_));
      const double delta_eqps = mimi::solvers::ScalarSolve(residual,
                                                           0.0,
                                                           lower_bound,
                                                           upper_bound,
                                                           opts,
                                                           q);
      // compute sqrt(3/2) * eta / norm(eta)
      // this term is use for both s and plastic strain
      // this is equivalent to
      // s = eta (see above)
      // 3/2 * s / q = 3/2 * s * sqrt(2/3) / norm(s)
      // didn't quite get why this is called Np yet,
      // but as this references serac's implementation
      // here it goes
      // we can directly incorperate this into s, but
      N_p.Set(1.5 / q, s);

      s.Add(-2.0 * G_ * delta_eqps, N_p);
      if (!state->freeze_) {
        accumulated_plastic_strain += delta_eqps;
        plastic_strain.Add(delta_eqps, N_p);
      }
    }

    // returning s + p * I
    mfem::Add(s, I_, p, sigma);
  }
};

struct JohnsonCookRateDependentHardening : public JohnsonCookHardening {
  using Base_ = JohnsonCookHardening;
  using ADScalar_ = Base_::ADScalar_;

  using Base_::A_;
  using Base_::B_;
  using Base_::n_;
  double C_;
  // several ways to call this
  // Jannis calls this reference strain rate
  // wikipedia call this:
  double effective_plastic_strain_rate_;

  /// @brief  this is a long name for a hardening model
  /// @return
  std::string Name() const { return "JohnsonCookRateDependentHardening"; }

  bool IsRateDependent() const { return true; }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain,
           const ADScalar_& equivalent_plastic_strain_rate) const {
    MIMI_FUNC()
    return Evaluate(accumulated_plastic_strain)
           * (1.0
              + (C_
                 * std::log(equivalent_plastic_strain_rate
                            / effective_plastic_strain_rate_)));
  }
};

/// @brief Computational Methods for plasticity p260, box 7.5
/// specialized for visco plasticity.
/// Then eta = s, instead of eta = s - beta
/// eta: relative stress
/// beta: backstress tensor
/// s: stress deviator
/// Implementation reference from serac
/// Considers nonlinear Isotropic hardening
///
///
class J2NonlinearVisco : public MaterialBase {
public:
  using Base_ = MaterialBase;
  using MaterialStatePtr_ = typename Base_::MaterialStatePtr_;
  using HardeningPtr_ = std::shared_ptr<HardeningBase>;
  using ADScalar_ = typename HardeningBase::ADScalar_;

  // additional parameters
  HardeningPtr_ hardening_;

  /// @brief Bulk Modulus (lambda + 2 / 3 mu)
  double K_; // K
  /// shear modulus (=mu)
  double G_;

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
  /// I am thread-safe. Don't touch me after Setup
  mfem::DenseMatrix I_;

  /// some constants
  const double sqrt_3_2_ = std::sqrt(3.0 / 2.0);

  /// lookup index for matrix
  /// elastic strain
  static constexpr const int k_eps{0};
  /// stress deviator
  static constexpr const int k_s{1};
  /// N_p
  static constexpr const int k_N_p{2};
  /// eps dot (plastic strain rate)
  static constexpr const int k_eps_dot{3};

public:
  virtual std::string Name() const { return "J2NonlinearVisco"; }

  /// answers if this material is suitable for visco-solids.
  virtual bool IsRateDependent() const { return true; }

  /// gives hint to integrator, which stress is implemented we need
  virtual bool UsesCauchy() const {
    MIMI_FUNC()
    return false;
  }

  virtual void Setup(const int dim, const int nthread) {
    MIMI_FUNC()

    /// base setup for conversions and dim, nthread
    Base_::Setup(dim, nthread);

    // check if this is an appropriate hardening.
    if (hardening_) {
      if (!hardening_->IsRateDependent()) {
        mimi::utils::PrintAndThrowError(hardening_->Name(),
                                        "is not rate-dependent.");
      }
    } else {
      mimi::utils::PrintAndThrowError("hardening missing for", Name());
    }

    // I
    I_.Diag(1., dim);

    // match variable name with the literature
    K_ = lambda_ + (2 * mu_ / 3);
    G_ = mu_;

    /// make space for du_dX
    aux_matrices_.resize(
        n_threads_,
        Vector_<mfem::DenseMatrix>(4, mfem::DenseMatrix(dim_)));
  }

  virtual MaterialStatePtr_ CreateState() const {
    MIMI_FUNC();

    std::shared_ptr<State> state = std::make_shared<State>();
    // create 2 matrices with the size of dim x dim and zero initialize
    state->matrices_.resize(state->k_state_matrices);
    for (mfem::DenseMatrix& mat : state->matrices_) {
      mat.SetSize(dim_, dim_);
      mat = 0.;
    }
    // one scalar, also zero
    state->scalars_.resize(state->k_state_scalars, 0.);
    return state;
  };

  virtual void EvaluateCauchy(const mfem::DenseMatrix& F,
                              const int& i_thread,
                              MaterialStatePtr_& state,
                              mfem::DenseMatrix& sigma) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("Invalid call for ", Name());
  }

  virtual void EvaluateCauchy(const mfem::DenseMatrix& F,
                              const mfem::DenseMatrix& F_dot,
                              const int& i_thread,
                              MaterialStatePtr_& state,
                              mfem::DenseMatrix& sigma) {
    MIMI_FUNC()

    // get aux
    auto& i_aux = aux_matrices_[i_thread];
    mfem::DenseMatrix& eps = i_aux[k_eps];
    mfem::DenseMatrix& s = i_aux[k_s];
    mfem::DenseMatrix& N_p = i_aux[k_N_p];
    mfem::DenseMatrix& eps_dot = i_aux[k_eps_dot];

    // get states
    mfem::DenseMatrix& plastic_strain =
        state->matrices_[State::k_plastic_strain];
    double& accumulated_plastic_strain =
        state->scalars_[State::k_accumulated_plastic_strain];

    // precompute aux values
    // eps, p, s, eta, q, phi
    ElasticStrain(F, plastic_strain, eps);
    const double p = K_ * eps.Trace();
    Dev(eps, dim_, 2.0 * G_, s);
    const double q = sqrt_3_2_ * Norm(s);

    // admissibility
    const double eqps_old = accumulated_plastic_strain;
    auto residual = [eqps_old, *this](auto delta_eqps,
                                      auto trial_mises) -> ADScalar_ {
      return trial_mises - 3.0 * G_ * delta_eqps
             - hardening_->Evaluate(eqps_old + delta_eqps);
    };

    const double tolerance = hardening_->SigmaY() * k_tol;

    if (residual(0.0, q) > tolerance) {
      /// return mapping
      mimi::solvers::ScalarSolverOptions opts{.xtol = 0.,
                                              .rtol = tolerance,
                                              .max_iter = 100};

      const double lower_bound = 0.0;
      const double upper_bound =
          (q - hardening_->Evaluate(eqps_old).GetValue() / (3.0 * G_));
      const double delta_eqps = mimi::solvers::ScalarSolve(residual,
                                                           0.0,
                                                           lower_bound,
                                                           upper_bound,
                                                           opts,
                                                           q);
      // compute sqrt(3/2) * eta / norm(eta)
      // this term is use for both s and plastic strain
      // this is equivalent to
      // s = eta (see above)
      // 3/2 * s / q = 3/2 * s * sqrt(2/3) / norm(s)
      // didn't quite get why this is called Np yet,
      // but as this references serac's implementation
      // here it goes
      // we can directly incorperate this into s, but
      N_p.Set(1.5 / q, s);

      s.Add(-2.0 * G_ * delta_eqps, N_p);
      if (!state->freeze_) {
        accumulated_plastic_strain += delta_eqps;
        plastic_strain.Add(delta_eqps, N_p);
      }
    }

    // returning s + p * I
    mfem::Add(s, I_, p, sigma);
  }
};

} // namespace mimi::integrators
