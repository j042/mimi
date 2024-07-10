// mimi
#include "mimi/py/py_ex2.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_ex2(py::module_& m) {
    m.def("calculate_volume_and_compliance",
          &mimi::py::calculate_volume_and_compliance,
          py::arg("mesh_file"),
          py::arg("output_dir"),
          py::arg("sim_name") = "Example2");
}

} // namespace mimi::py
