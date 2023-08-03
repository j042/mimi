#include <pybind11/pybind11.h>

namespace mimi {
namespace py = pybind11;

void init_pysolids(py::module_&);
} // namespace mimi

PYBIND11_MODULE(mimi, m) { mimi::init_pysolids(m); }