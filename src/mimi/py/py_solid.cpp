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
      .def("nurbs", &PySolid::GetNurbs)
      .def("add_spline",
           &PySolid::AddSpline,
           py::arg("spline_name"),
           py::arg("spline"))
      .def("setup", &PySolid::Setup)
      .def_property_readonly("current_time", &PySolid::CurrentTime)
      .def_property("time_step_size",
                    &PySolid::GetTimeStepSize,
                    &PySolid::SetTimeStepSize)
      .def_readwrite("runtime_communication", &PySolid::runtime_communication_)
      .def("linear_form_view2", &PySolid::LinearFormView2, py::arg("lf_name"))
      .def("solution_view",
           &PySolid::SolutionView,
           py::arg("fe_space_name"),
           py::arg("component_name"))
      .def("boundary_dof_ids",
           &PySolid::BoundaryDofIds,
           py::arg("fe_space_name"),
           py::arg("bid"),
           py::arg("dim"))
      .def("zero_dof_ids", &PySolid::ZeroDofIds, py::arg("fe_space_name"))
      .def("nonlinear_from2", &PySolid::NonlinearForm2, py::arg("nlf_name"))
      .def("step_time2", &PySolid::StepTime2)
      .def("configure_newton",
           &PySolid::ConfigureNewton,
           py::arg("name"),
           py::arg("rel_tol"),
           py::arg("abs_tol"),
           py::arg("max_iter"),
           py::arg("iterative_mode"))
      .def("newton_final_norms", &PySolid::NewtonFinalNorms)
      .def("update_contact_lagrange", &PySolid::UpdateContactLagrange)
      .def("fill_contact_lagrange",
           &PySolid::FillContactLagrange,
           py::arg("value"))
      .def("fixed_point_solve2", &PySolid::FixedPointSolve2)
      .def("fixed_point_advance2",
           py::overload_cast<>(&PySolid::FixedPointAdvance2))
      .def("fixed_point_alm_solve2",
           &PySolid::FixedPointALMSolve2,
           py::arg("n_outer"),
           py::arg("n_inner"),
           py::arg("n_final"),
           py::arg("final_penalty_scale"),
           py::arg("rel_tol"),
           py::arg("abs_tol"),
           py::arg("gap_tol"),
           py::arg("restart_augmentation"))
      .def("fixed_point_advanced_vector_views",
           &PySolid::FixedPointAdvancedVectorViews)
      .def("advance_time2", &PySolid::AdvanceTime2);
}

} // namespace mimi::py
