// mimi
#include "mimi/py/py_stokes.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_incompressible_fluid(py::module_& m) {
  py::class_<PyIncompressibleFluid,
             std::shared_ptr<PyIncompressibleFluid>,
             PySolid>
      klasse(m, "IncompressibleFluid");
  klasse.def(py::init<>())
      .def("set_material",
           &PyIncompresibleFluid::SetMaterial,
           py::arg("material"));
}

} // namespace mimi::py
