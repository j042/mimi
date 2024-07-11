// mimi
#include "mimi/py/py_le_cantilever_beam.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_le_cantilever_beam(py::module_& m) {
  py::class_<PyLECantileverBeam, std::shared_ptr<PyLECantileverBeam>> klasse(m, "LECantileverBeam");
  klasse.def(py::init<std::string&, std::string&, std::string&>(), py::arg("mesh_file"), py::arg("output_dir"), py::arg("sim_name"))
      .def("solve", &PyLECantileverBeam::solve)
      .def_readonly("compliance", &PyLECantileverBeam::compliance)
      .def_readonly("volume", &PyLECantileverBeam::volume)
      .def("get_solution", &PyLECantileverBeam::get_solution);
}

} // namespace mimi::py
