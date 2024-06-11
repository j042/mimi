// mimi
#include "mimi/py/py_linear_elasticity.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_linear_elasticity(py::module_& m) {
  py::class_<PyLinearElasticity, std::shared_ptr<PyLinearElasticity>, PySolid>
      klasse(m, "LinearElasticity");
  klasse.def(py::init<>())
      .def("set_parameters",
           &PyLinearElasticity::SetParameters_,
           py::arg("young"),
           py::arg("poisson"),
           py::arg("density"),
           py::arg("viscosity") = -1.0);
}

} // namespace mimi::py
