#include <memory>

#include <pybind11/pybind11.h>

#include "mimi/integrators/materials.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_material(py::module_& m) {

  using MaterialBase = mimi::integrators::MaterialBase;

  py::class_<MaterialBase, std::shared_ptr<MaterialBase>> klasse(m,
                                                                 "PyMaterial");
  klasse.def(py::init<>())
      .def("name", &MaterialBase::Name)
      .def_readwrite("density", &MaterialBase::density_)
      .def_readwrite("viscosity", &MaterialBase::viscosity_);
}

} // namespace mimi::py
