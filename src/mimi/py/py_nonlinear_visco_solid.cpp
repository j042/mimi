// mimi
#include "mimi/py/py_nonlinear_visco_solid.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_nonlinear_visco_solid(py::module_& m) {
  py::class_<PyNonlinearViscoSolid,
             std::shared_ptr<PyNonlinearViscoSolid>,
             PySolid>
      klasse(m, "PyNonlinearViscoSolid");
  klasse.def(py::init<>())
      .def("set_material", &PyNonlinearSolid::SetMaterial, py::arg("material"));
}

} // namespace mimi::py
