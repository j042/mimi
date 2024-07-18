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
  using JCHardening = mimi::materials::JohnsonCookHardening;
  using JCViscoHardening = mimi::materials::JohnsonCookRateDependentHardening;
  using JCConstTemperatureHardening =
      mimi::materials::JohnsonCookConstantTemperatureHardening;
  using JCThermoViscoHardening =
      mimi::materials::JohnsonCookAdiabaticRateDependentHardening;

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

  py::class_<JCHardening, std::shared_ptr<JCHardening>, HardeningBase> jc_h(
      m,
      "JohnsonCookHardening");
  jc_h.def(py::init<>())
      .def_readwrite("A", &JCHardening::A_)
      .def_readwrite("B", &JCHardening::B_)
      .def_readwrite("n", &JCHardening::n_);

  py::class_<JCConstTemperatureHardening,
             std::shared_ptr<JCConstTemperatureHardening>,
             JCViscoHardening>
      jc_visco_const_temp(m, "JohnsonCookViscoConstantTemperatureHardening");
  jc_visco_const_temp.def(py::init<>())
      .def_readwrite("reference_temperature",
                     &JCConstTemperatureHardening::reference_temperature_)
      .def_readwrite("melting_temperature",
                     &JCConstTemperatureHardening::melting_temperature_)
      .def_readwrite("m", &JCConstTemperatureHardening::m_)
      .def_readwrite("temperature", &JCConstTemperatureHardening::temperature_);

  py::class_<JCThermoViscoHardening,
             std::shared_ptr<JCThermoViscoHardening>,
             JCViscoHardening>
      jc_thermo_visco(m, "JohnsonCookThermoViscoHardening");
  jc_thermo_visco.def(py::init<>())
      .def_readwrite("reference_temperature",
                     &JCThermoViscoHardening::reference_temperature_)
      .def_readwrite("melting_temperature",
                     &JCThermoViscoHardening::melting_temperature_)
      .def_readwrite("m", &JCThermoViscoHardening::m_);
}

} // namespace mimi::py
