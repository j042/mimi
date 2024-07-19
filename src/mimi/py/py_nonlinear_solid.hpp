#pragma once

#include <memory>

#include <mfem.hpp>

#include "mimi/integrators/mortar_contact.hpp"
#include "mimi/integrators/nonlinear_solid.hpp"
#include "mimi/materials/materials.hpp"
#include "mimi/operators/nonlinear_solid.hpp"
#include "mimi/py/py_solid.hpp"

namespace mimi::py {

class PyNonlinearSolid : public PySolid {
public:
  using Base_ = PySolid;
  using MaterialPointer_ = std::shared_ptr<mimi::materials::MaterialBase>;

  MaterialPointer_ material_;

  PyNonlinearSolid() = default;

  virtual void SetMaterial(const MaterialPointer_& material) {
    MIMI_FUNC()
    material_ = material;
    mimi::utils::PrintInfo("Set material", material_->Name());
  }

  virtual void Setup(const int nthreads = -1);
};

} // namespace mimi::py
