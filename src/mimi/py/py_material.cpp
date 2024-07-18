#include <memory>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include "mimi/materials/material_utils.hpp"
#include "mimi/materials/materials.hpp"

namespace mimi::py {

namespace py = pybind11;

template<typename T>
T* Ptr(const py::array_t<T>& arr) {
  return static_cast<T*>(arr.request().ptr);
}

void init_py_material(py::module_& m) {
  /// material laws
  using MaterialBase = mimi::materials::MaterialBase;
  using StVK = mimi::materials::StVenantKirchhoff;
  using CompOgdenNH = mimi::materials::CompressibleOgdenNeoHookean;
  using J2 = mimi::materials::J2Linear;
  using J2 = mimi::materials::J2Nonlinear;
  using J2Simo = mimi::materials::J2Simo;
  using J2Log = mimi::materials::J2Log;

  /// input type
  using ADScalar = typename HardeningBase::ADScalar_;

  py::class_<MaterialBase, std::shared_ptr<MaterialBase>> klasse(m, "Material");

  klasse.def(py::init<>())
      .def("name", &MaterialBase::Name)
      .def_readwrite("density", &MaterialBase::density_)
      .def_readwrite("viscosity", &MaterialBase::viscosity_)
      .def("set_young_poisson", &MaterialBase::SetYoungPoisson)
      .def("set_lame", &MaterialBase::SetLame);

  py::class_<StVK, std::shared_ptr<StVK>, MaterialBase> stvk(
      m,
      "StVenantKirchhoff");
  stvk.def(py::init<>());

  py::class_<CompOgdenNH, std::shared_ptr<CompOgdenNH>, MaterialBase> conh(
      m,
      "CompressibleOgdenNeoHookean");
  conh.def(py::init<>());

  py::class_<J2Linear, std::shared_ptr<J2Linear>, MaterialBase> j2_lin(
      m,
      "J2Linear");
  j2_lin.def(py::init<>())
      .def_readwrite("isotropic_hardening", &J2Linear::isotropic_hardening_)
      .def_readwrite("kinematic_hardening", &J2Linear::kinematic_hardening_)
      .def_readwrite("sigma_y", &J2Linear::sigma_y_);

  py::class_<J2, std::shared_ptr<J2>, MaterialBase> j2(m, "J2");
  j2.def(py::init<>())
      .def_readwrite("hardening", &J2::hardening_)
      .def_readwrite("heat_fraction", &J2::heat_fraction_)
      .def_readwrite("specific_heat", &J2::specific_heat_)
      .def_readwrite("initial_temperature", &J2::initial_temperature_)
      .def_readwrite("melting_temperature", &J2::melting_temperature_);

  py::class_<J2Simo, std::shared_ptr<J2Simo>, MaterialBase> j2_simo(m,
                                                                    "J2Simo");
  j2_simo.def(py::init<>())
      .def_readwrite("hardening", &J2Simo::hardening_)
      .def_readwrite("heat_fraction", &J2Simo::heat_fraction_)
      .def_readwrite("specific_heat", &J2Simo::specific_heat_)
      .def_readwrite("initial_temperature", &J2Simo::initial_temperature_)
      .def_readwrite("melting_temperature", &J2Simo::melting_temperature_);

  py::class_<J2Log, std::shared_ptr<J2Log>, MaterialBase> j2_log(m, "J2Log");
  j2_log.def(py::init<>())
      .def_readwrite("hardening", &J2Log::hardening_)
      .def_readwrite("heat_fraction", &J2Log::heat_fraction_)
      .def_readwrite("specific_heat", &J2Log::specific_heat_)
      .def_readwrite("initial_temperature", &J2Log::initial_temperature_)
      .def_readwrite("melting_temperature", &J2Log::melting_temperature_);
}

} // namespace mimi::py
