#include <pybind11/pybind11.h>

#include "mimi/coefficients/nearest_distance.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_nearest_distance(py::module_& m) {

  using NearestDistance = mimi::coefficients::NearestDistanceBase;
  using NearestDistanceToSplines = mimi::coefficients::NearestDistanceToSplines;

  py::class_<NearestDistance, std::shared_ptr<NearestDistance>> nearest_klasse(m, "PyNearestDistance");
    nearest_klasse.def(py::init<>())
    .def_readwrite("coefficient", &NearestDistance::coefficient_)
    .def_readwrite("tolerance", &NearestDistance::tolerance_);

  py::class_<NearestDistanceToSplines, std::shared_ptr<NearestDistanceToSplines>, NearestDistance>
    nearest_spline_klasse(m, "PyNearestDistanceToSplines");
    nearest_spline_klasse.def(py::init<>())
    .def("add_spline", &NearestDistanceToSplines::AddSpline, py::arg("spline"))
    .def("plant_kd_tree", &NearestDistanceToSplines::PlantKdTree, py::arg("resolution"), py::arg("nthreads") = 1);
}

} // namespace mimi::py