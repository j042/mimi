#include "pybind11/stl.h"

// mimi
#include "mimi/py/py_solid.hpp"

namespace mimi::py {
namespace py = pybind11;

void init_py_solid(py::module_& m) {
  py::class_<PySolid, std::shared_ptr<PySolid>> klasse(m, "Solid");

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
      .def("dof_map", &PySolid::DofMap, py::arg("fe_space"))
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
      .def("fixed_point_solve2", &PySolid::FixedPointSolve2)
      .def("fixed_point_advance2",
           py::overload_cast<>(&PySolid::FixedPointAdvance2))
      .def("fixed_point_advanced_vector_views",
           &PySolid::FixedPointAdvancedVectorViews)
      .def("advance_time2", &PySolid::AdvanceTime2);
}

void PySolid::ReadMesh(const std::string fname) {
  MIMI_FUNC()

  const char* fname_char = fname.c_str();

  // read mesh and save
  mesh_ = std::make_unique<mfem::Mesh>(fname_char, 1, 1);

  // nurbs check
  if (mesh_->NURBSext) {
    mimi::utils::PrintDebug("read NURBS mesh");
  } else {
    mimi::utils::PrintAndThrowError(fname, "Does not contain NURBS mesh.");
  }

  // generate boundary markers
  const int max_bdr_id = mesh_->bdr_attributes.Max();
  mimi::utils::PrintInfo("Maximum boundary id for this mesh is:", max_bdr_id);
  boundary_markers_.resize(max_bdr_id);
  for (int i{}; i < max_bdr_id; ++i) {
    auto& marker = boundary_markers_[i];
    marker.SetSize(max_bdr_id);
    marker = 0;
    marker[i] = 1;
  }
}

void PySolid::SaveMesh(const std::string fname) const {
  MIMI_FUNC()

  std::ofstream mesh_ofs(fname);
  mesh_ofs.precision(12);
  mesh_->Print(mesh_ofs);
}

std::unique_ptr<mfem::Mesh>& PySolid::Mesh() {
  MIMI_FUNC()

  if (!mesh_) {
    mimi::utils::PrintAndThrowError("Mesh not set.");
  }

  return mesh_;
}

const std::unique_ptr<mfem::Mesh>& PySolid::Mesh() {
  MIMI_FUNC()

  if (!mesh_) {
    mimi::utils::PrintAndThrowError("Mesh not set.");
  }

  return mesh_;
}

std::vector<int> PySolid::MeshDegrees() const {
  MIMI_FUNC()

  std::vector<int> degrees;
  degrees.reserve(MeshDim());

  for (const auto& d : Mesh()->NURBSext->GetOrders()) {
    degrees.push_back(d);
  }

  return degrees;
}

std::shared_ptr<mimi::utils::RuntimeCommunication>
PySolid::RuntimeCommunication() {
  MIMI_FUNC()
  if (!runtime_communication_) {
    runtime_communication_ =
        std::make_shared<mimi::utils::RuntimeCommunication>();
  }
  return runtime_communication_;
}

void PySolid::ElevateDegrees(const int degrees, const int max_degrees) {
  MIMI_FUNC()

  mimi::utils::PrintDebug("degrees input:", degrees);
  mimi::utils::PrintDebug("max_degrees set to", max_degrees);

  if (degrees > 0) {
    Mesh()->DegreeElevate(degrees, max_degrees);
  } else {
    mimi::utils::PrintWarning(degrees, "is invalid input. Skipping.");
  }

  // FYI
  auto ds = MeshDegrees();
  mimi::utils::PrintDebug("current degrees:");
  for (int i{}; i < MeshDim(); ++i) {
    mimi::utils::PrintDebug("dim", i, ":", ds[i]);
  }
}

void PySolid::Subdivide(const int n_subdivision) {
  MIMI_FUNC()

  mimi::utils::PrintDebug("n_subdivision:", n_subdivision);
  mimi::utils::PrintDebug("Number of elements before subdivision: ",
                          NumberOfElements());

  if (n_subdivision > 0) {
    for (int i{}; i < n_subdivision; ++i) {
      Mesh()->UniformRefinement();
    }
  }

  mimi::utils::PrintDebug("Number of elements after subdivision: ",
                          NumberOfElements());
}

void PySolid::FindBoundaryDofIds() {
  MIMI_FUNC()

  // find all true dof ids
  for (auto& [key, fes] : fe_spaces_) {
    mimi::utils::PrintDebug("Finding boundary dofs for", key, "FE Space.");

    const int max_bdr_id = fes.fe_space->GetMesh()->bdr_attributes.Max();

    // loop each bdr.
    for (int i{}; i < max_bdr_id; ++i) {
      // fespace's dim
      for (int j{}; j < fes.fe_space->GetVDim(); ++j) {
        // mark only bdr id for this loop
        mfem::Array<int> bdr_id_query(max_bdr_id);
        bdr_id_query = 0;    // clear
        bdr_id_query[i] = 1; // mark

        // query
        fes.fe_space->GetEssentialTrueDofs(bdr_id_query,
                                           fes.boundary_dof_ids[i][j],
                                           j);
      }
    }
  }

  // find dirichlet bcs
  for (auto& [name, fes] : fe_spaces_) {
    for (auto const& [bid, dims] :
         boundary_conditions_->InitialConfiguration().dirichlet_) {
      for (auto const& dim : dims) {

        mimi::utils::PrintDebug("For FE Space",
                                name,
                                "- finding boundary dofs for initial "
                                "configuration dirichlet bcs.",
                                "bid:",
                                bid,
                                "dim:",
                                dim);

        // append saved dofs
        // may have duplicating dofs, harmless.
        fes.zero_dofs.Append(fes.boundary_dof_ids[bid][dim]);
      }
    }
    // on second thought, it is a bit harmless.
    fes.zero_dofs.Sort();
    fes.zero_dofs.Unique();
  }
}

py::dict PySolid::GetNurbs() {
  MIMI_FUNC()

  py::dict nurbs;
  // degrees
  nurbs["degrees"] = py::cast(MeshDegrees());

  // knot vectors
  py::list kvs;
  for (int i{}; i < MeshDim(); ++i) {
    const auto& mfem_kv = *Mesh()->NURBSext->GetKnotVector(i);

    py::list kv;
    for (int j{}; j < mfem_kv.Size(); ++j) {
      kv.append(mfem_kv[j]);
    }

    kvs.append(kv);
  }
  nurbs["knot_vectors"] = kvs;

  // control points
  mfem::Vector cps(NumberOfVertices() * MeshDim());
  // mimi::utils::PrintInfo("num patch:",Mesh()->NURBSext->GetNP());
  Mesh()->GetNodes(cps);
  // Mesh()->NURBSext->SetCoordsFromPatches(cps);
  nurbs["control_points"] =
      mimi::py::NumpyCopy<double>(cps, cps.Size() / MeshDim(), MeshDim());

  /*
  mimi::utils::PrintInfo("num patch:",Mesh()->NURBSext->GetNP());

  mfem::NURBSPatchMap pm(Mesh()->NURBSext);
  std::vector<const mfem::KnotVector*> kv(2);
  kv[0] = Mesh()->NURBSext->GetKnotVector(0);
  kv[1] = Mesh()->NURBSext->GetKnotVector(1);
  pm.SetPatchDofMap(0, kv.data());

  for (int i{}; i < kv[0]->GetNCP(); ++i) {
    for (int j{}; j < kv[1]->GetNCP(); ++j) {
      mimi::utils::PrintInfo(" ", i, j, pm(i, j));
    }
  }

  for (int i{}; i < kv[0]->GetNCP(); ++i) {
    for (int j{}; j < kv[1]->GetNCP(); ++j) {
      mimi::utils::PrintInfo(" ", j, i, pm(j, i));
    }
  }
  */

  // weights
  mfem::Vector& ws = Mesh()->NURBSext->GetWeights();
  nurbs["weights"] = mimi::py::NumpyCopy<double>(ws, ws.Size(), 1);

  return nurbs;
}

void PySolid::AddSpline(std::string const& s_name,
                        std::shared_ptr<splinepy::py::PySpline> spline) {
  MIMI_FUNC()

  splines_[s_name] = spline;
  mimi::utils::PrintInfo("I got a spline", spline->WhatAmI());
}

void PySolid::SetupNTheads(const int n_threads) {
  MIMI_FUNC()

  if (n_threads < 1) {
    mimi::utils::PrintAndThrowError("nthreads can't be smaller than 1.");
  }

#ifdef MIMI_USE_OMP
  omp_set_num_threads(n_threads);
  mimi::utils::PrintInfo("Using OPENMP.",
                         "Max threads:",
                         omp_get_max_threads(),
                         "Num threads:",
                         omp_get_num_threads());
#endif
}

py::array_t<int> PySolid::DofMap(const std::string& key) const {
  MIMI_FUNC()
  mfem::NURBSExtension& ext = *fe_spaces_.at(key).fe_space->GetNURBSext();
  const int dim = MeshDim();
  const int n_dof = Mesh()->GetNodes()->Size() / dim;

  py::array_t<int> dofmap(n_dof);
  int* dm_d = Ptr(dofmap);
  for (int i{}; i < n_dof; ++i) {
    dm_d[i] = ext.DofMap(i);
  }
  return dofmap;
}

void PySolid::ConfigureNewton(const std::string& name,
                              const double rel_tol,
                              const double abs_tol,
                              const double max_iter,
                              const bool iterative_mode) {
  MIMI_FUNC()

  auto& newton = newton_solvers_.at(name);
  newton->SetRelTol(rel_tol);
  newton->SetAbsTol(abs_tol);
  newton->SetMaxIter(max_iter);
  newton->iterative_mode = iterative_mode;
}

void PySolid::SetDynamicSystem2(mfem::SecondOrderTimeDependentOperator* oper2,
                                mimi::solvers::OdeBase* ode2) {

  MIMI_FUNC()

  oper2_ = std::unique_ptr<mfem::SecondOrderTimeDependentOperator>(oper2);
  ode2_solver_ = std::unique_ptr<mimi::solvers::OdeBase>(ode2);

  // ode solvers also wants to know dirichlet dofs
  auto* op_base = dynamic_cast<mimi::operators::OperatorBase*>(oper2_.get());
  ode2_solver_->SetupDirichletDofs(op_base->dirichlet_dofs_);

  RuntimeCommunication()->InitializeTimeStep();
}

py::array_t<double> PySolid::LinearFormView2(const std::string lf_name) {
  MIMI_FUNC()

  auto* op_base = dynamic_cast<mimi::operators::OperatorBase*>(oper2_.get());
  assert(op_base);

  auto& lf = op_base->linear_forms_.at(lf_name); // seems to not raise
  if (!lf) {
    mimi::utils::PrintAndThrowError("Requested linear form -",
                                    lf_name,
                                    "- does not exist.");
  }

  return mimi::py::NumpyView<double>(*lf, lf->Size());
}

py::array_t<double> PySolid::SolutionView(const std::string& fes_name,
                                          const std::string& component_name) {
  MIMI_FUNC()

  auto& fes = fe_spaces_.at(fes_name);
  auto& grid_func = fes.grid_functions.at(component_name);

  return mimi::py::NumpyView<double>(grid_func,
                                     grid_func.Size()); // will be raveled.
}

py::array_t<int> PySolid::BoundaryDofIds(const std::string& fes_name,
                                         const int& bid,
                                         const int& dim) {
  MIMI_FUNC()

  auto& fes = fe_spaces_.at(fes_name);
  auto& mfem_array = fes.boundary_dof_ids.at(bid).at(dim);

  return mimi::py::NumpyCopy<int>(mfem_array, mfem_array.Size());
}

py::array_t<int> PySolid::ZeroDofIds(const std::string& fes_name) {
  MIMI_FUNC()

  auto& fes = fe_spaces_.at(fes_name);

  return mimi::py::NumpyCopy<int>(fes.zero_dofs, fes.zero_dofs.Size());
}

std::shared_ptr<mimi::forms::Nonlinear>
PySolid::NonlinearForm2(const std::string& nlf_name) {
  MIMI_FUNC()

  assert(oper2_);

  auto* mimi_oper2 = dynamic_cast<mimi::operators::OperatorBase*>(oper2_.get());

  if (!mimi_oper2) {
    mimi::utils::PrintAndThrowError(
        "2nd order dynamic system does not exist yet.");
  }

  return mimi_oper2->nonlinear_forms_.at(nlf_name);
}

void PySolid::StepTime2() {
  MIMI_FUNC()

  assert(x2_);
  assert(x2_dot_);
  mimi::utils::PrintInfo("🌲🌲 StepTime2 🌲🌲 - t:", t_, "dt:", dt_);

  ode2_solver_->StepTime2(*x2_, *x2_dot_, t_, dt_);
  auto& rc = *RuntimeCommunication();
  if (rc.ShouldSave("x")) {
    rc.SaveDynamicVector("x_", *x2_);
  }
  if (rc.ShouldSave("v")) {
    rc.SaveDynamicVector("v_", *x2_dot_);
  }
  rc.NextTimeStep(dt_);
}

void PySolid::FixedPointSolve2() {
  MIMI_FUNC()

  assert(x2_);
  assert(x2_dot_);
  mimi::utils::PrintInfo("📌📌 FixedPointSolve2 📌📌 - t:", t_, "dt:", dt_);

  ode2_solver_->FixedPointSolve2(*x2_, *x2_dot_, t_, dt_);
}

void PySolid::FixedPointAdvance2(mfem::Vector& fp_x, mfem::Vector& fp_v) {
  MIMI_FUNC()

  const int x_size = x2_->Size();

  fp_x.SetSize(x_size);
  fp_v.SetSize(x_size);

  double* fx = fp_x.GetData();
  double* fv = fp_v.GetData();
  const double* x = x2_->GetData();
  const double* v = x2_dot_->GetData();
  for (int i{}; i < x_size; ++i) {
    fx[i] = x[i];
    fv[i] = v[i];
  }

  mimi::utils::PrintInfo("📍📍 FixedPointAdvance2 📍📍 - t:", t_, "dt:", dt_);
  ode2_solver_->FixedPointAdvance2(fp_x, fp_v, t_, dt_);
}

py::tuple PySolid::FixedPointAdvance2() {
  MIMI_FUNC()

  FixedPointAdvance2(fixed_point_advanced_x_, fixed_point_advanced_v_);

  return FixedPointAdvancedVectorViews();
}

py::tuple PySolid::FixedPointAdvancedVectorViews() {
  MIMI_FUNC()

  return py::make_tuple(
      NumpyView<double>(fixed_point_advanced_x_,
                        fixed_point_advanced_x_.Size() / MeshDim(),
                        MeshDim()),
      NumpyView<double>(fixed_point_advanced_v_,
                        fixed_point_advanced_v_.Size() / MeshDim(),
                        MeshDim()));
}

void PySolid::AdvanceTime2() {
  MIMI_FUNC()

  assert(x2_);
  assert(x2_dot_);
  mimi::utils::PrintInfo("🚂🚂 AdvanceTime2 🚂🚂 - t:", t_, "dt:", dt_);

  ode2_solver_->AdvanceTime2(*x2_, *x2_dot_, t_, dt_);
  auto& rc = *RuntimeCommunication();
  if (rc.ShouldSave("x")) {
    rc.SaveDynamicVector("x_", *x2_);
  }
  if (rc.ShouldSave("v")) {
    rc.SaveDynamicVector("v_", *x2_dot_);
  }
  rc.NextTimeStep(dt_);
}

} // namespace mimi::py
