// mimi
#include "mimi/py/py_nonlinear_solid.hpp"

// we need to initialize static variable in non-header file.
// keep it here until we separate cpp
namespace mimi::integrators {
bool MaterialState::freeze_ = false;
}

namespace mimi::py {

namespace py = pybind11;

void init_py_nonlinear_solid(py::module_& m) {
  py::class_<PyNonlinearSolid, std::shared_ptr<PyNonlinearSolid>, PySolid>
      klasse(m, "PyNonlinearSolid");
  klasse.def(py::init<>())
      .def("set_material", &PyNonlinearSolid::SetMaterial, py::arg("material"));
}

} // namespace mimi::py
