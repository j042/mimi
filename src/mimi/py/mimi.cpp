#include <pybind11/pybind11.h>

namespace mimi::py {
namespace py = pybind11;

void init_py_solid(py::module_&);
void init_py_boundary_conditions(py::module_&);
} // namespace mimi::py

PYBIND11_MODULE(mimi, m) {
  mimi::py::init_py_solid(m);
  mimi::py::init_py_boundary_conditions(m);
}
