/* pybind11 */
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

// mimi
#include "mimi/py/py_utils.hpp"
#include "mimi/utils/ad.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_ad(py::module_& m) {
  using AD = mimi::utils::ADScalar<double, 0>;
  using AD_DT = typename AD::DerivType_;

  using ADVec = mimi::utils::ADVector<0>;

  py::class_<AD> ad(m, "PyAD");
  ad.def(py::init<const double&, const int&>())
      .def("v", &AD::GetValue)
      .def("d",
           [](const AD& a) -> py::array_t<double> {
             auto d = a.GetDerivatives();
             return NumpyCopy<double>(d, d.size());
           })
      .def("activate", &AD::SetActiveComponent, py::arg("i"))
      .def(py::self + py::self)
      .def(py::self + double())
      .def(double() + py::self)
      .def(py::self - py::self)
      .def(py::self - double())
      .def(double() - py::self)
      .def(py::self * py::self)
      .def(py::self * double())
      .def(double() * py::self)
      .def(py::self / py::self)
      .def(py::self / double())
      .def(double() / py::self);

  py::class_<ADVec> ad_vec(m, "PyADVec");
  ad_vec.def(py::init<int>());
}

} // namespace mimi::py
