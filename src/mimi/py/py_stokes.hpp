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
  // we create one more mesh that
  std::unique_ptr<mfem::Mesh> vel_mesh_;

  // we work with a mono system instead of block structure
  std::shared_ptr<mimi::utils::FluidPrecomputedData> fluid_precomputed_data_;

  mfem::BlockVector block_v_p_;

  mfem::Array<int> zero_dofs_;

  PyStokes() = default;

  virtual void SetMaterial(const MaterialPointer_& material) {
    MIMI_FUNC()
    material_ = material;
    mimi::utils::PrintInfo("Set material", material_->Name());
  }

  virtual void Setup(const int nthreads = -1);
};

} // namespace mimi::py
