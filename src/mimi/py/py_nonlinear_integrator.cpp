// mimi
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/integrators/penalty_contact.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_nonlinear_base_integrator(py::module_& m) {
  using mimi::integrators::NonlinearBase;
  py::class_<NonlinearBase, std::shared_ptr<NonlinearBase>> klasse(
      m,
      "PyNonlinearIntegratorBase");
  // klasse.def(py::init<>())
  klasse.def("name", &NonlinearBase::Name)
      .def("gap_norm", &NonlinearBase::LastGapNorm)
      .def("temperature",
           [](NonlinearBase& nlb,
              const py::array_t<double>& x,
              py::array_t<double>& temperature) {
             mfem::Vector x_vec(static_cast<double*>(x.request().ptr),
                                x.size());
             mfem::Vector t_vec(static_cast<double*>(temperature.request().ptr),
                                temperature.size());

             nlb.Temperature(x_vec, t_vec);
           })
      .def("accumulated_plastic_strain",
           [](NonlinearBase& nlb,
              const py::array_t<double>& x,
              py::array_t<double>& temperature) {
             mfem::Vector x_vec(static_cast<double*>(x.request().ptr),
                                x.size());
             mfem::Vector t_vec(static_cast<double*>(temperature.request().ptr),
                                temperature.size());

             nlb.AccumulatedPlasticStrain(x_vec, t_vec);
           });
}

} // namespace mimi::py
