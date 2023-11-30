#pragma once

#include <mfem.hpp>

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
                              MaterialMaterialStatePtr_& state,
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

  /// @brief unless implemented, this will try to call evaluate sigma and
  /// transform if none of stress is implemented, you will be stuck in a
  /// neverending loop current implementation is not so memory efficient
  virtual void EvaluatePK1(const mfem::DenseMatrix& F,
                           const int& i_thread,
                           MaterialState& state,
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
/// mu / det(F) * (B - I) + lambda * (det(F))
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

  /// mu / det(F) * (B - I) + lambda * (det(F))
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

} // namespace mimi::integrators
