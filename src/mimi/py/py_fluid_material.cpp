#include <memory>

#include <pybind11/pybind11.h>

#include "mimi/materials/fluid_materials.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_fluid_material(py::module_& m) {
  /// material laws
  using MaterialBase = mimi::materials::FluidMaterialBase;

  py::class_<MaterialBase, std::shared_ptr<MaterialBase>> klasse(
      m,
      "FluidMaterial");

  klasse.def(py::init<>())
      .def("name", &MaterialBase::Name)
      .def_readwrite("density", &MaterialBase::density_)
      .def_readwrite("viscosity", &MaterialBase::viscosity_);
}

} // namespace mimi::py
