#pragma once

#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/materials/fluid_materials.hpp"

namespace mimi::integrators {
class Stokes : public NonlinearBase {
protected:
  std::shared_ptr<mimi::utils::FluidMatrialBase> material_;
  std::shared_ptr<mimi::utils::FluidPrecomputedData> precomputed_;

public:
  Stokes(const std::string& name,
         const std::shared_ptr<mimi::utils::FluidMatrialBase> mat,
         const std::shared_ptr<mimi::utils::FluidPrecomputedData>& precomputed)
      : NonlinearBase(name, nullptr),
        material_(mat),
        precomputed_(precomputed) {}

  virtual void Prepare() {}
  virtual void AddDomainResidual(const mfem::BlockVector& current_sol,
                                 mfem::Vector& residual) {}
  virtual void AddDomainResidualAndGrad(const mfem::BlockVector& current_sol,
                                        const double grad_factor,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) {}
};

} // namespace mimi::integrators
