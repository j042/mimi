// mimi
#include "mimi/py/py_solid.hpp"

namespace mimi::py {
namespace py = pybind11;

void init_py_solid(py::module_& m) {
  py::class_ <PySolid, std::shared_ptr<PySolid>> klass(m, "PySolid");

  klasse.def(py::init<>())
    .def("read_mesh", &PySolid::ReadMesh, py::arg("fname"))
    .def("mesh_dim", &PySolid::MeshDim)
    .def("mesh_degrees", &PySolid::MeshDegrees)
    .def("n_vertices", &PySolid::NumberOfVertices)
    .def("n_elements", &PySolid::NumberOfElements)
    .def("n_boundary_elements", &PySolid::NumberofBoundaryElements)
    .def("n_subelements", &PySolid::NumberOfSubelements)
    .def("elevate_degrees", &PySolid::ElevateDegrees, py::args("degrees"), py::args("max_degrees")=50)
    .def("subdivide", &PySolid::Subdivide, py::args("n_subdivision"))
    .def("boundary_condition", &PySolid::GetBoundaryCondition, &PySolid::SetBoundaryCondition)
    .def("setup", &PySolid::Setup)
}

} // namespace mimi::py
