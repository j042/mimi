// mimi
#include "mimi/py/py_utils.hpp"
#include "mimi/solvers/ode.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_ode(py::module_& m) {

  using OdeBase = mimi::solvers::OdeBase;
  using GeneralizedAlpha2 = mimi::solvers::GeneralizedAlpha2;
  using AverageAcceleration = mimi::solvers::AverageAcceleration;
  using HHTAlpha = mimi::solvers::HHTAlpha;
  using WBZAlpha = mimi::solvers::WBZAlpha;
  using Newmark = mimi::solvers::Newmark;
  using LinearAcceleration = mimi::solvers::LinearAcceleration;
  using CenteralDifference = mimi::solvers::CentralDifference;
  using FoxGoodwin = mimi::solvers::FoxGoodwin;

  py::class_<OdeBase, std::shared_ptr<OdeBase>> klasse(m, "PyOde");
  klasse.def(py::init<>())
      .def("acceleration",
           [](OdeBase& ob) {
             mfem::Vector* acc = ob.Acceleration();
             if (!acc) {
               mimi::utils::PrintAndThrowError(
                   "Acceeleration vector does not exist.");
               return nullptr;
             }

             return mimi::utils::NumpyView(*acc, acc->Size());
           })
      .def("name", &OdeBase::Name);

  py::class_<GeneralizedAlpha2, std::shared_ptr<GeneralizedAlpha2>, OdeBase>
      gen_alpha(m, "PyGeneralizedAlpha2");
  py::class_<AverageAcceleration, std::shared_ptr<AverageAcceleration>, OdeBase>
      aa(m, "PyAverageAcceleration");
  py::class_<HHTAlpha, std::shared_ptr<HHTAlpha>, OdeBase> htta(m,
                                                                "PyHHTAlpha");
  py::class_<WBZAlpha, std::shared_ptr<WBZAlpha>, OdeBase> wbza(m,
                                                                "PyWBZAlpha");
  py::class_<Newmark, std::shared_ptr<Newmark>, OdeBase> newmark(m,
                                                                 "PyNewmark");
  py::class_<LinearAcceleration, std::shared_ptr<LinearAcceleration>, OdeBase>
      la(m, "PyLinearAcceleration");
  py::class_<CenteralDifference, std::shared_ptr<CenteralDifference>, OdeBase>
      cd(m, "PyCenteralDifference");
  py::class_<FoxGoodwin, std::shared_ptr<FoxGoodwin>, OdeBase> fg(
      m,
      "PyFoxGoodwin");
}

} // namespace mimi::py
