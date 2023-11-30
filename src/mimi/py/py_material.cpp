#include <memory>

#include <pybind11/pybind11.h>

#include "mimi/integrators/materials.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_material(py::module_& m) {

  using MaterialBase = mimi::integrators::MaterialBase;
  using StVK = mimi::integrators::StVenantKirchhoff;
  using CompOgdenNH = mimi::integrators::CompressibleOgdenNeoHookean;

  py::class_<MaterialBase, std::shared_ptr<MaterialBase>> klasse(m,
                                                                 "PyMaterial");
  klasse.def(py::init<>())
      .def("name", &MaterialBase::Name)
      .def_readwrite("density", &MaterialBase::density_)
      .def_readwrite("viscosity", &MaterialBase::viscosity_)
      .def_readwrite("lambda_", &MaterialBase::lambda_)
      .def_readwrite("mu", &MaterialBase::mu_)
      .def("uses_cauchy", &MaterialBase::UsesCauchy);

  py::class_<StVK, std::shared_ptr<StVK>, MaterialBase> stvk(
      m,
      "PyStVenantKirchhoff");
  stvk.def(py::init<>());

  py::class_<CompOgdenNH, std::shared_ptr<CompOgdenNH>, MaterialBase> conh(
      m,
      "PyCompressibleOdgenNeoHookean");
  conh.def(py::init<>());
}

} // namespace mimi::py
