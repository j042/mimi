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
  using J2NonlinVisco = mimi::integrators::J2NonlinearVisco;

  /// hardening laws
  using HardeningBase = mimi::integrators::HardeningBase;
  using PowerLawHardening = mimi::integrators::PowerLawHardening;
  using VoceHardening = mimi::integrators::VoceHardening;
  using JCHardening = mimi::integrators::JohnsonCookHardening;
  using JCViscoHardening = mimi::integrators::JohnsonCookRateDependentHardening;

  /// input type
  using ADScalar = typename HardeningBase::ADScalar_;

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
  h_base.def(py::init<>())
      .def("sigma_y", &HardeningBase::SigmaY)
      .def("name", &HardeningBase::Name)
      .def("is_rate_dependent", &HardeningBase::IsRateDependent)
      .def("evaluate",
           [](const HardeningBase& self, const double eqps) -> double {
             return self.Evaluate(ADScalar(eqps)).GetValue();
           })
      .def("visco_evaluate",
           [](const HardeningBase& self,
              const double eqps,
              const double eqps_dot) -> double {
             return self.Evaluate(ADScalar(eqps), eqps_dot).GetValue();
           });

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

  py::class_<JCViscoHardening, std::shared_ptr<JCViscoHardening>, JCHardening>
      jc_visco(m, "PyJohnsonCookViscoHardening");
  jc_visco.def(py::init<>())
      .def_readwrite("C", &JCViscoHardening::C_)
      .def_readwrite("eps_dot0",
                     &JCViscoHardening::effective_plastic_strain_rate_);

  py::class_<J2NonlinVisco, std::shared_ptr<J2NonlinVisco>, MaterialBase>
      j2_visco(m, "PyJ2ViscoIsotropicHardening");
  j2_visco.def(py::init<>())
      .def_readwrite("hardening", &J2NonlinVisco::hardening_);
}

} // namespace mimi::py
