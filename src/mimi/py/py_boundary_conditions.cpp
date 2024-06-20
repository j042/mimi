/* pybind11 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// mimi
#include "mimi/utils/boundary_conditions.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_boundary_conditions(py::module_& m) {
  using BC = mimi::utils::BoundaryConditions;
  using BCM = typename BC::BCMarker;
  py::class_<BCM> marker(m, "BoundaryMarker");
  marker
      .def("dirichlet",
           &BCM::Dirichlet,
           py::arg("bid"),
           py::arg("dim"),
           py::return_value_policy::reference_internal)
      .def("pressure",
           &BCM::Pressure,
           py::arg("bid"),
           py::arg("value"),
           py::return_value_policy::reference_internal)
      .def("traction",
           &BCM::Traction,
           py::arg("bid"),
           py::arg("dim"),
           py::arg("value"),
           py::return_value_policy::reference_internal)
      .def("body_force",
           &BCM::BodyForce,
           py::arg("dim"),
           py::arg("value"),
           py::return_value_policy::reference_internal)
      .def("constant_velocity",
           &BCM::ConstantVelocity,
           py::arg("bid"),
           py::arg("dim"),
           py::arg("value"),
           py::return_value_policy::reference_internal)
      .def("contact",
           &BCM::Contact,
           py::arg("bid"),
           py::arg("nearest_distance"),
           py::return_value_policy::reference_internal);

  py::class_<BC, std::shared_ptr<BC>> bc(m, "BoundaryConditions");
  bc.def(py::init<>())
      .def("print", [](BC& bc) { bc.Print(); })
      .def_property_readonly("initial",
                             &BC::InitialConfiguration,
                             py::return_value_policy::reference_internal)
      .def_property_readonly("current",
                             &BC::CurrentConfiguration,
                             py::return_value_policy::reference_internal);
}

} // namespace mimi::py
