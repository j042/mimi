#pragma once

#include <memory>

#include <mfem.hpp>

#include "mimi/integrators/stokes.hpp"
#include "mimi/materials/fluid_materials.hpp"
#include "mimi/operators/incompressible_fluid.hpp"
#include "mimi/py/py_solid.hpp"
#include "mimi/utils/precomputed.hpp"

namespace mimi::py {

class PyStokes : public PySolid {
public:
  using Base_ = PySolid;
  using MaterialPointer_ = std::shared_ptr<mimi::materials::FluidMaterialBase>;

  MaterialPointer_ material_;
  // we create one more mesh for velocity which is one order higher; for
  // Taylor-Hood elements
  std::unique_ptr<mfem::Mesh> vel_mesh_;

  // we work with a mono system instead of block structure
  std::shared_ptr<mimi::utils::FluidPrecomputedData> fluid_precomputed_data_;

  // blockvector for velocity and pressure FE spaces
  mfem::BlockVector block_v_p_;

  // true dofs of velocity and pressure
  mfem::Array<int> zero_dofs_;

  PyStokes() = default;

  /// @brief Sets material properties
  /// @param material Pointer to material properties
  virtual void SetMaterial(const MaterialPointer_& material) {
    MIMI_FUNC()
    material_ = material;
    mimi::utils::PrintInfo("Set material", material_->Name());
  }

  /// @brief Setup meshes, FE spaces, precomputed data, integrators, forms,
  /// solvers and operators
  /// @param nthreads Number of threads
  virtual void Setup(const int nthreads = -1);

  /// @brief Solve stationary Stokes system
  virtual void StaticSolve() {
    MIMI_FUNC()
    mfem::Vector zero;
    Base_::newton_solvers_.at("stokes")->Mult(zero, block_v_p_);
  }
};

} // namespace mimi::py