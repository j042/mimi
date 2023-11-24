#pragma once

#include <mfem.hpp>

#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

template<typename T>
using Vector_ = mimi::utils::Vector<T>

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
protected:
  int dim_;
  int n_threads_;

public:
  double density_{-1.0};
  double viscosity_{-1.0};
  double lambda_{-1.0};
  double mu_{-1.0};

  virtual void Prepare(const int dim, const int n_threads_) { MIMI_FUNC() }

  virtual std::string Name() const { return "MaterialBase"; }

  /// self setup. will be called once.
  /// most of the time Probably will set up I
  virtual void Setup(const int dim) { MIMI_FUNC() }

  /// integrators will put each initialize state to this function
  /// you can use this to initialize
  virtual void InitializeState(const int dim,
                               MaterialState& state){MIMI_FUNC()};

  virtual void EvaluateStress(const mfem::DenseMatrix& F,
                              MaterialState& state,
                              mfem::DenseMatrix& stress) const {
    MIMI_FUNC()
    // here, depends on material, we can either call Cauchy stress or PK1
    // as a base nothing.
    // once we implement some sort of material, we can make this abstract
  }

  /// give hint to integrator, which integration we need
  virtual bool PhysicalIntegration() const {
    MIMI_FUNC()
    return false;
  }
  virtual void EvaluateCauchyStress(const mfem::DenseMatrix& F,
                                    MaterialState& state,
                                    mfem::DenseMatrix& sigma) const {
    MIMI_FUNC()
  }
  virtual void EvaluatePK1(const mfem::DenseMatrix& F,
                           MaterialState& state,
                           mfem::DenseMatrix& p) const {
    MIMI_FUNC()
  }
  virtual void EvaluateGrad() const { MIMI_FUNC() }
};

class StVenantKirchhoff : public MaterialBase {
protected:
  mfem::DenseMatrix I;

public:
  virtual std::string Name() const { return "StVenantKirchhoff"; }

  virtual void Setup(const int dim) {
    MIMI_FUNC()
    I.Diag(1., dim);
  }

  virtual void InitializeState(const int dim, MaterialState& state) {
    MIMI_FUNC()
    // for StVK, we need 3 temporary matrix -> C, E, S
    state.matrices_.resize(3, mfem::DenseMatrix(dim, dim));
  };

  virtual void EvaluateStress(const mfem::DenseMatrix& F,
                              MaterialState& state,
                              mfem::DenseMatrix& stress) const {
    MIMI_FUNC()
    EvaluatePK1(F, state, stress);
  }

  virtual void EvaluatePK1(const mfem::DenseMatrix& F,
                           MaterialState&,
                           mfem::DenseMatrix& P) const {
    MIMI_FUNC()
    const int size = F.Height();
    mfem::DenseMatrix I;
    I.Diag(1, size);

    mfem::DenseMatrix C(size);
    mfem::DenseMatrix E(size);
    mfem::DenseMatrix S(size);

    // C
    mfem::MultAtB(F, F, C);

    // E
    mfem::Add(.5, C, -.5, I, E);

    // S
    mfem::Add(lambda_ * E.Trace(), I, 2 * mu_, E, S);

    // P
    mfem::Mult(F, S, P);
  }
};

} // namespace mimi::integrators
