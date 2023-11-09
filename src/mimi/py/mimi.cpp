#include <pybind11/pybind11.h>

namespace mimi::py {
namespace py = pybind11;
void init_py_boundary_conditions(py::module_&);
void init_py_solid(py::module_&);
void init_py_linear_elasticity(py::module_& m);
void init_py_nearest_distance(py::module_& m);
} // namespace mimi::py

PYBIND11_MODULE(mimi, m) {
  mimi::py::init_py_boundary_conditions(m);
  mimi::py::init_py_solid(m);
  mimi::py::init_py_linear_elasticity(m);
  mimi::py::init_py_nearest_distance(m);
}
