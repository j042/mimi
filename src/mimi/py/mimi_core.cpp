#include <pybind11/pybind11.h>

namespace mimi::py {
namespace py = pybind11;
void init_py_boundary_conditions(py::module_&);
void init_py_solid(py::module_&);
void init_py_nearest_distance(py::module_&);
void init_py_material(py::module_&);
void init_py_hardening(py::module_&);
void init_py_nonlinear_solid(py::module_&);
void init_py_ad(py::module_&);
void init_py_runtime_communication(py::module_&);
void init_py_le_cantilever_beam(py::module_&);
} // namespace mimi::py

PYBIND11_MODULE(mimi_core, m) {
  mimi::py::init_py_boundary_conditions(m);
  mimi::py::init_py_solid(m);
  mimi::py::init_py_nearest_distance(m);
  mimi::py::init_py_material(m);
  mimi::py::init_py_hardening(m);
  mimi::py::init_py_nonlinear_solid(m);
  mimi::py::init_py_ad(m);
  mimi::py::init_py_runtime_communication(m);
  mimi::py::init_py_le_cantilever_beam(m);
}
