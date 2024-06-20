// mimi
#include "mimi/py/py_nonlinear_solid.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_nonlinear_solid(py::module_& m) {
  py::class_<PyNonlinearSolid, std::shared_ptr<PyNonlinearSolid>, PySolid>
      klasse(m, "NonlinearSolid");
  klasse.def(py::init<>())
      .def("set_material", &PyNonlinearSolid::SetMaterial, py::arg("material"));
}

} // namespace mimi::py
