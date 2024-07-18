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
  using J2 = mimi::materials::J2;
  using J2NonlinIso = mimi::materials::J2NonlinearIsotropic;
  using J2NonlinVisco = mimi::materials::J2NonlinearVisco;
  using J2NonlinAdiabaticVisco = mimi::materials::J2AdiabaticVisco;
  using J2NonlinAdiabaticViscoLarge = mimi::materials::J2AdiabaticViscoLarge;
  using J2NonlinAdiabaticViscoLog = mimi::materials::J2AdiabaticViscoLogStrain;

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

  py::class_<J2, std::shared_ptr<J2>, MaterialBase> j2(m, "J2");
  j2.def(py::init<>())
      .def_readwrite("isotropic_hardening", &J2::isotropic_hardening_)
      .def_readwrite("kinematic_hardening", &J2::kinematic_hardening_)
      .def_readwrite("sigma_y", &J2::sigma_y_);

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

  py::class_<J2NonlinIso, std::shared_ptr<J2NonlinIso>, MaterialBase> j2_nl_hi(
      m,
      "J2NonlinearIsotropic");
  j2_nl_hi.def(py::init<>())
      .def_readwrite("hardening", &J2NonlinIso::hardening_);

  py::class_<JCViscoHardening, std::shared_ptr<JCViscoHardening>, JCHardening>
      jc_visco(m, "JohnsonCookViscoHardening");
  jc_visco.def(py::init<>())
      .def_readwrite("C", &JCViscoHardening::C_)
      .def_readwrite("eps0_dot",
                     &JCViscoHardening::effective_plastic_strain_rate_);

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

  py::class_<J2NonlinVisco, std::shared_ptr<J2NonlinVisco>, MaterialBase>
      j2_visco(m, "J2ViscoIsotropic");
  j2_visco.def(py::init<>())
      .def_readwrite("hardening", &J2NonlinVisco::hardening_);

  py::class_<J2NonlinAdiabaticVisco,
             std::shared_ptr<J2NonlinAdiabaticVisco>,
             MaterialBase>
      j2_adia_visco(m, "J2AdiabaticViscoIsotropic");
  j2_adia_visco.def(py::init<>())
      .def_readwrite("hardening", &J2NonlinAdiabaticVisco::hardening_)
      .def_readwrite("heat_fraction", &J2NonlinAdiabaticVisco::heat_fraction_)
      .def_readwrite("specific_heat", &J2NonlinAdiabaticVisco::specific_heat_)
      .def_readwrite("initial_temperature",
                     &J2NonlinAdiabaticVisco::initial_temperature_);

  py::class_<J2NonlinAdiabaticViscoLarge,
             std::shared_ptr<J2NonlinAdiabaticViscoLarge>,
             MaterialBase>
      j2_av_large(m, "J2AdiabaticViscoLarge");
  j2_av_large.def(py::init<>())
      .def_readwrite("hardening", &J2NonlinAdiabaticViscoLarge::hardening_)
      .def_readwrite("heat_fraction",
                     &J2NonlinAdiabaticViscoLarge::heat_fraction_)
      .def_readwrite("specific_heat",
                     &J2NonlinAdiabaticViscoLarge::specific_heat_)
      .def_readwrite("initial_temperature",
                     &J2NonlinAdiabaticViscoLarge::initial_temperature_);

  py::class_<J2NonlinAdiabaticViscoLog,
             std::shared_ptr<J2NonlinAdiabaticViscoLog>,
             J2NonlinAdiabaticVisco>
      j2_av_log(m, "J2LogStrainAdiabaticVisco");
  j2_av_log.def(py::init<>());

  m.def("eigen_and_adat", [](py::array_t<double>& arr) {
    mfem::DenseMatrix mat(static_cast<double*>(arr.request().ptr),
                          arr.shape(0),
                          arr.shape(1));
    py::array_t<double> e_val(arr.shape(0));
    py::array_t<double> e_vec(arr.size());
    py::array_t<double> adat(arr.size());
    mfem::DenseMatrix adatmat, e_vecmat;
    mfem::Vector e_valvec;
    e_valvec.SetDataAndSize(static_cast<double*>(e_val.request().ptr),
                            e_val.size());
    adatmat.UseExternalData(static_cast<double*>(adat.request().ptr),
                            arr.shape(0),
                            arr.shape(1));
    e_vecmat.UseExternalData(static_cast<double*>(e_vec.request().ptr),
                             arr.shape(0),
                             arr.shape(1));

    mat.CalcEigenvalues(static_cast<double*>(e_val.request().ptr),
                        static_cast<double*>(e_vec.request().ptr));

    mfem::MultADAt(e_vecmat, e_valvec, adatmat);
    return py::make_tuple(e_val, e_vec, adat);
  });

  m.def("log_strain",
        [](const py::array_t<double>& F, const py::array_t<double>& state) {
          py::array_t<double> out(F.size());
          const int d0{(int) F.shape(0)}, d1{(int) F.shape(1)};
          mimi::integrators::NonlinearSolidWorkData tmp;
          tmp.aux_mat_.assign(2, mfem::DenseMatrix(d0, d0));
          tmp.aux_vec_.assign(1, mfem::Vector(d0));
          mfem::DenseMatrix& state_mat = tmp.stress_; // use any matrix
          mfem::DenseMatrix& out_mat = tmp.F_dot_;
          tmp.SetDim(d0);
          tmp.F_.UseExternalData(Ptr(F), d0, d1);
          state_mat.UseExternalData(Ptr(state), d0, d1);
          out_mat.UseExternalData(Ptr(out), d0, d1);
          mimi::materials::LogarithmicStrain<0, 1, 0>(state_mat, tmp, out_mat);
          return out;
        });
}

} // namespace mimi::py
