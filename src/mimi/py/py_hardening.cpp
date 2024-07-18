#include <memory>

#include <pybind11/pybind11.h>

#include "mimi/materials/material_hardening.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_hardening(py::module_& m) {
  /// hardening laws
  using HardeningBase = mimi::materials::HardeningBase;
  using PowerLawHardening = mimi::materials::PowerLawHardening;
  using VoceHardening = mimi::materials::VoceHardening;
  using JC = mimi::materials::JohnsonCookHardening;
  using JC_R = mimi::materials::JohnsonCookRateDependentHardening;
  using JC_RT =
      mimi::materials::JohnsonCookTemperatureAndRateDependentHardening;
  using JC_RConstT = mimi::materials::JohnsonCookConstantTemperatureHardening;

  /// input type
  using ADScalar = typename HardeningBase::ADScalar_;

  py::class_<HardeningBase, std::shared_ptr<HardeningBase>> h_base(m,
                                                                   "Hardening");
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
      power_law_h(m, "PowerLawHardening");
  power_law_h.def(py::init<>())
      .def_readwrite("sigma_y", &PowerLawHardening::sigma_y_)
      .def_readwrite("n", &PowerLawHardening::n_)
      .def_readwrite("eps0", &PowerLawHardening::eps0_);

  py::class_<VoceHardening, std::shared_ptr<VoceHardening>, HardeningBase>
      voce_h(m, "VoceHardening");
  voce_h.def(py::init<>())
      .def_readwrite("sigma_y", &VoceHardening::sigma_y_)
      .def_readwrite("sigma_sat", &VoceHardening::sigma_sat_)
      .def_readwrite("strain_constant", &VoceHardening::strain_constant_);

  py::class_<JC, std::shared_ptr<JC>, HardeningBase> jc_h(
      m,
      "JohnsonCookHardening");
  jc_h.def(py::init<>())
      .def_readwrite("A", &JC::A_)
      .def_readwrite("B", &JC::B_)
      .def_readwrite("n", &JC::n_);

  py::class_<JC_R, std::shared_ptr<JC_R>, JC> jc_r(
      m,
      "JohnsonCookRateDependentHardening");
  jc_r.def(py::init<>())
      .def_readwrite("effective_plastic_strain_rate",
                     &JC_R::effective_plastic_strain_rate_);

  py::class_<JC_RT, std::shared_ptr<JC_RT>, JC_R> jc_rt(
      m,
      "JohnsonCookTemperatureAndRateDependentHardening");
  jc_rt.def(py::init<>())
      .def_readwrite("reference_temperature", &JC_RT::reference_temperature_)
      .def_readwrite("m", &JC_RT::m_);

  py::class_<JC_RConstT, std::shared_ptr<JC_RConstT>, JC_RT>
      jc_visco_const_temp(m, "JohnsonCookViscoConstantTemperatureHardening");
  jc_visco_const_temp.def(py::init<>());
}

} // namespace mimi::py
