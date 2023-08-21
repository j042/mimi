#include "pybind11/stl.h"

// mimi
#include "mimi/py/py_solid.hpp"

namespace mimi::py {
namespace py = pybind11;

void init_py_solid(py::module_& m) {
  py::class_<PySolid, std::shared_ptr<PySolid>> klasse(m, "PySolid");

  klasse.def(py::init<>())
      .def("read_mesh", &PySolid::ReadMesh, py::arg("fname"))
      .def("save_mesh", &PySolid::SaveMesh, py::arg("fname"))
      .def("mesh_dim", &PySolid::MeshDim)
      .def("mesh_degrees", &PySolid::MeshDegrees)
      .def("n_vertices", &PySolid::NumberOfVertices)
      .def("n_elements", &PySolid::NumberOfElements)
      .def("n_boundary_elements", &PySolid::NumberOfBoundaryElements)
      .def("n_subelements", &PySolid::NumberOfSubelements)
      .def("elevate_degrees",
           &PySolid::ElevateDegrees,
           py::arg("degrees"),
           py::arg("max_degrees") = 50)
      .def("subdivide", &PySolid::Subdivide, py::arg("n_subdivision"))
      .def_property("boundary_condition",
                    &PySolid::GetBoundaryConditions,
                    &PySolid::SetBoundaryConditions)
      .def("setup", &PySolid::Setup);
}

} // namespace mimi::py
