#include <memory>

#include <pybind11/pybind11.h>

#include "mimi/integrators/materials.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_material(py::module_& m) {
  /// material laws
  using MaterialBase = mimi::integrators::MaterialBase;
  using StVK = mimi::integrators::StVenantKirchhoff;
  using CompOgdenNH = mimi::integrators::CompressibleOgdenNeoHookean;
  using J2 = mimi::integrators::J2;
  using J2NonlinHi = mimi::integrators::J2NonlinearIsotropicHardening;

  /// hardening laws
  using HardeningBase = mimi::integrators::HardeningBase;
  using PowerLawHardening = mimi::integrators::PowerLawHardening;
  using VoceHardening = mimi::integrators::VoceHardening;
  using JCHardening = mimi::integrators::JohnsonCookHardening;

  py::class_<MaterialBase, std::shared_ptr<MaterialBase>> klasse(m,
                                                                 "PyMaterial");

  klasse.def(py::init<>())
      .def("name", &MaterialBase::Name)
      .def_readwrite("density", &MaterialBase::density_)
      .def_readwrite("viscosity", &MaterialBase::viscosity_)
      //.def_readwrite("lambda_", &MaterialBase::lambda_)
      //.def_readwrite("mu", &MaterialBase::mu_)
      //.def_readwrite("young", &MaterialBase::young_)
      //.def_readwrite("poisson", &MaterialBase::poisson_)
      .def("uses_cauchy", &MaterialBase::UsesCauchy)
      .def("set_young_poisson", &MaterialBase::set_young_poisson)
      .def("set_lame", &MaterialBase::set_lame);

  py::class_<StVK, std::shared_ptr<StVK>, MaterialBase> stvk(
      m,
      "PyStVenantKirchhoff");
  stvk.def(py::init<>());

  py::class_<CompOgdenNH, std::shared_ptr<CompOgdenNH>, MaterialBase> conh(
      m,
      "PyCompressibleOgdenNeoHookean");
  conh.def(py::init<>());

  py::class_<J2, std::shared_ptr<J2>, MaterialBase> j2(m, "PyJ2");
  j2.def(py::init<>())
      .def_readwrite("isotropic_hardening", &J2::isotropic_hardening_)
      .def_readwrite("kinematic_hardening", &J2::kinematic_hardening_)
      .def_readwrite("sigma_y", &J2::sigma_y_);

  py::class_<HardeningBase, std::shared_ptr<HardeningBase>> h_base(
      m,
      "PyHardening");
  h_base.def(py::init<>()).def("sigma_y", &HardeningBase::SigmaY);

  py::class_<PowerLawHardening,
             std::shared_ptr<PowerLawHardening>,
             HardeningBase>
      power_law_h(m, "PyPowerLawHardening");
  power_law_h.def(py::init<>())
      .def_readwrite("sigma_y", &PowerLawHardening::sigma_y_)
      .def_readwrite("n", &PowerLawHardening::n_)
      .def_readwrite("eps0", &PowerLawHardening::eps0_);

  py::class_<VoceHardening, std::shared_ptr<VoceHardening>, HardeningBase>
      voce_h(m, "PyVoceHardening");
  voce_h.def(py::init<>())
      .def_readwrite("sigma_y", &VoceHardening::sigma_y_)
      .def_readwrite("sigma_sat", &VoceHardening::sigma_sat_)
      .def_readwrite("strain_constant", &VoceHardening::strain_constant_);

  py::class_<JCHardening, std::shared_ptr<JCHardening>, HardeningBase> jc_h(
      m,
      "PyJohnsonCookHardening");
  jc_h.def(py::init<>())
      .def_readwrite("A", &JCHardening::A_)
      .def_readwrite("B", &JCHardening::B_)
      .def_readwrite("n", &JCHardening::n_);

  py::class_<J2NonlinHi, std::shared_ptr<J2NonlinHi>, MaterialBase> j2_nl_hi(
      m,
      "PyJ2NonlinearIsotropicHardening");
  j2_nl_hi.def(py::init<>())
      .def_readwrite("hardening", &J2NonlinHi::hardening_);
}

} // namespace mimi::py
