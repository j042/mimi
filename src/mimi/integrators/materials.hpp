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
  virtual const std::string& Name() const { return "MaterialBase"; }

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

} // namespace mimi::integrators
