#include <memory>

/* pybind11 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // cast std::vectors

// mimi
#include "mimi/utils/runtime_communication.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_runtime_communication(py::module_& m) {
  using RC = mimi::utils::RuntimeCommunication;

  py::class_<RC, std::shared_ptr<RC>> rc(m, "RuntimeCommunication");
  rc.def(py::init<>())
      .def_readwrite("fname", &RC::fname_)
      .def("set_fname", &RC::SetFName)
      .def("get_real", &RC::GetReal)
      .def("set_real", &RC::SetReal)
      .def("get_int", &RC::GetInteger)
      .def("set_int", &RC::SetInteger)
      .def("append_should_save", &RC::AppendShouldSave)
      .def("should_save", &RC::ShouldSave)
      .def("setup_real_history", &RC::SetupRealHistory)
      .def("get_real_history", &RC::GetRealHistory)
      .def("get_real_history_at", &RC::GetRealHistoryAt)
      .def("save_real_history", &RC::SaveRealHistory)
      .def("latest_vector", &RC::LatestVector);
}

} // namespace mimi::py
