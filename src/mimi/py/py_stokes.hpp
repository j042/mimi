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

  virtual void ReadMesh(const std::string fname) {
    Base_::ReadMesh(fname);
    std::swap(mesh_, vel_mesh_);
    Base_::ReadMesh(fname);
    std::swap(mesh_, vel_mesh_);
    mimi::utils::PrintDebug("Elevating velocity mesh");
    VelMesh()->DegreeElevate(1, 50);
  }

  virtual std::unique_ptr<mfem::Mesh>& VelMesh() {
    if (!vel_mesh_) {
      mimi::utils::PrintAndThrowError("No vel mesh.");
    }
    return vel_mesh_;
  }

  virtual void ElevateDegrees(const int degrees, const int max_degrees = 50) {
    mimi::utils::PrintDebug("degrees input:", degrees);
    mimi::utils::PrintDebug("max_degrees set to", max_degrees);

    if (degrees > 0) {
      Mesh()->DegreeElevate(degrees, max_degrees);
      VelMesh()->DegreeElevate(degrees, max_degrees);
    } else {
      mimi::utils::PrintWarning(degrees, "is invalid input. Skipping.");
    }

    // FYI
    auto ds = MeshDegrees();
    mimi::utils::PrintDebug("current degrees:");
    for (int i{}; i < MeshDim(); ++i) {
      mimi::utils::PrintDebug("dim", i, ":", ds[i]);
    }
  }

  virtual void Subdivide(const int n_subdivision) {
    MIMI_FUNC()

    mimi::utils::PrintDebug("n_subdivision:", n_subdivision);
    mimi::utils::PrintDebug("Number of elements before subdivision: ",
                            NumberOfElements());

    if (n_subdivision > 0) {
      for (int i{}; i < n_subdivision; ++i) {
        Mesh()->UniformRefinement();
        VelMesh()->UniformRefinement();
      }
    }

    mimi::utils::PrintDebug("Number of elements after subdivision: ",
                            NumberOfElements());
  }

  virtual void SetMaterial(const MaterialPointer_& material) {
    MIMI_FUNC()
    material_ = material;
    mimi::utils::PrintInfo("Set material", material_->Name());
  }

  py::dict GetVelocityNurbs() {
    MIMI_FUNC()

    std::swap(mesh_, vel_mesh_);
    py::dict n_dict = GetNurbs();
    std::swap(mesh_, vel_mesh_);
    return n_dict;
  }

  virtual void Setup(const int nthreads = -1);

  virtual void StaticSolve() {
    MIMI_FUNC()
    mfem::Vector zero;
    Base_::newton_solvers_.at("stokes")->Mult(zero, block_v_p_);
  }
};

} // namespace mimi::py
