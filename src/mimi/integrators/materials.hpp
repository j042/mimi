#pragma once

#include <mfem.hpp>

#include "mimi/utils/print.hpp"

namespace mimi::integrators {

/// saves material state at quadrature point
struct MaterialState {};

/// Material base.
/// We can either define material or material state (in this case one material)
/// at each quad point. Something to consider before implementing everything
class MaterialBase {
public:
  double density_{-1.0};
  double viscosity_{-1.0};
  double lambda_{-1.0};
  double mu_{-1.0};
  virtual std::string Name() const { return "MaterialBase"; }

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
public:
  virtual std::string Name() const { return "StVenantKirchhoff"; }

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
