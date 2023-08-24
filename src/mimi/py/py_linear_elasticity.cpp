// mimi
#include "mimi/py/py_linear_elasticity.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_linear_elasticity(py::module_& m) {
  py::class_<PyLinearElasticity, std::shared_ptr<PyLinearElasticity>, PySolid> klasse(m, "PyLinearElasticity");
  klasse.def(py::init<>());
}

} // namespace mimi::py
