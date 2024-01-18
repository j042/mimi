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
      .def("gap_norm", &NonlinearBase::GapNorm);
}

} // namespace mimi::py
