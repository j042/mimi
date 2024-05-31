#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

/* pybind11 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// splinepy
#include <splinepy/py/py_spline.hpp>

// mimi
#include "mimi/forms/nonlinear.hpp"
#include "mimi/operators/base.hpp"
#include "mimi/py/py_utils.hpp"
#include "mimi/solvers/newton.hpp"
#include "mimi/solvers/ode.hpp"
#include "mimi/utils/boundary_conditions.hpp"
#include "mimi/utils/precomputed.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::py {

namespace py = pybind11;

class PySolid {
protected:
  // second order time dependent systems
  std::unique_ptr<mimi::solvers::OdeBase> ode2_solver_ = nullptr;
  std::unique_ptr<mfem::SecondOrderTimeDependentOperator> oper2_ = nullptr;
  // non-owning ptr to solution vectors.
  // you don't need to set this if you plan to implement time stepping yourself.
  // used in PySolid's base time stepping methods
  mfem::GridFunction* x2_;
  mfem::GridFunction* x2_dot_;

  // first order time dependent systems
  std::unique_ptr<mimi::solvers::OdeBase> ode1_solver_ = nullptr;
  std::unique_ptr<mfem::TimeDependentOperator> oper1_ = nullptr;
  // non-owning ptr to solution vectors.
  // you don't need to set this if you plan to implement time stepping yourself.
  // used in PySolid's base time stepping methods
  mfem::GridFunction* x1_;

  // solvers
  std::map<std::string, std::shared_ptr<mimi::solvers::LineSearchNewton>>
      newton_solvers_;
  std::map<std::string, std::shared_ptr<mfem::Solver>> linear_solvers_;

  // mesh
  std::unique_ptr<mfem::Mesh> mesh_ = nullptr;

  // boundary markers. nothing more than array filled with zeros except the
  // marker
  mimi::utils::Vector<mfem::Array<int>> boundary_markers_;

  struct FESpace {
    std::string name_{"None"};
    std::unique_ptr<mfem::FiniteElementSpace> fe_space_{nullptr};
    std::map<int, std::map<int, mfem::Array<int>>> boundary_dof_ids_;
    std::unordered_map<std::string, mfem::GridFunction> grid_functions_;
    mfem::Array<int> zero_dofs_;
    std::unordered_map<std::string,
                       std::shared_ptr<mimi::utils::PrecomputedData>>
        precomputed_;
  };

  // there can be multiple fe spaces
  std::unordered_map<std::string, FESpace> fe_spaces_{};

  // bc manager
  std::shared_ptr<mimi::utils::BoundaryConditions> boundary_conditions_ =
      nullptr;

  // fsi coupling
  // a raw rhs vector for fsi loads
  std::shared_ptr<mfem::Vector> rhs_vector_ = nullptr;
  mfem::Vector fixed_point_advanced_x_;
  mfem::Vector fixed_point_advanced_v_;

  // frequently accessed pointers for augmented lagrange loops
  struct {
    std::shared_ptr<mimi::forms::Nonlinear> contact_form;
    std::shared_ptr<mimi::solvers::LineSearchNewton> newton_solver;
    mimi::utils::Vector<double> scene_coeffs;

    void SaveSceneCoefficients() {
      assert(contact_form);
      scene_coeffs.clear();
      scene_coeffs.reserve(contact_form->boundary_face_nfi_.size());
      for (auto& contact_integ : contact_form->boundary_face_nfi_) {
        scene_coeffs.push_back(contact_integ->PenaltyFactor());
      }
    }

    void UpdateContactLagrange() {
      MIMI_FUNC()

      assert(contact_form);
      for (auto& contact_integ : contact_form->boundary_face_nfi_) {
        contact_integ->UpdateLagrange();
      }
    }

    void FillContactLagrange(const double val) {
      MIMI_FUNC()

      assert(contact_form);
      for (auto& contact_integ : contact_form->boundary_face_nfi_) {
        contact_integ->FillLagrange(value);
      }
    }

    double GapNorm(const mfem::Vector& test_x) const {
      MIMI_FUNC()

      double total{};
      for (auto& contact_integ : contact_form->boundary_face_nfi_) {
        total += contact_integ->GapNorm(test_x, -1);
      }
      return total;
    }

    bool Converged(const double gap, const double gap_tol) const {
      MIMI_FUNC()

      return newton_solver->GetConverged() && gap < gap_tol;
    }
  } ALM;

  // holder for coefficients
  std::map<std::string, std::shared_ptr<mfem::Coefficient>> coefficients_;

  // holder for vector coefficients
  std::map<std::string, std::shared_ptr<mfem::VectorCoefficient>>
      vector_coefficients_;

  // holder for splines
  std::unordered_map<std::string, std::shared_ptr<splinepy::py::PySpline>>
      splines_;

  // current time, CET
  double t_{0.0};
  double dt_{0.0};

public:
  PySolid() = default;
  virtual ~PySolid() = default;

  /// @brief sets mesh
  /// @param fname
  virtual void ReadMesh(const std::string fname) {
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

  virtual void SaveMesh(const std::string fname) const {
    MIMI_FUNC()

    std::ofstream mesh_ofs(fname);
    mesh_ofs.precision(12);
    mesh_->Print(mesh_ofs);
  }

  /// @brief returns mesh. If it's missing, it will raise.
  /// @return
  virtual std::unique_ptr<mfem::Mesh>& Mesh() {
    MIMI_FUNC()

    if (!mesh_) {
      mimi::utils::PrintAndThrowError("Mesh not set.");
    }

    return mesh_;
  }

  /// @brief returns mesh. If it's missing, it will raise.
  /// @return
  virtual const std::unique_ptr<mfem::Mesh>& Mesh() const {
    MIMI_FUNC()

    if (!mesh_) {
      mimi::utils::PrintAndThrowError("Mesh not set.");
    }

    return mesh_;
  }

  /// @brief returns mesh dim (geometry dim)
  /// @return
  virtual int MeshDim() const {
    MIMI_FUNC()

    return Mesh()->Dimension();
  }

  /// @brief degrees of mesh.
  /// @return std::vector<int>, but will be casted fo py::list
  virtual std::vector<int> MeshDegrees() const {
    MIMI_FUNC()

    std::vector<int> degrees;
    degrees.reserve(MeshDim());

    for (const auto& d : Mesh()->NURBSext->GetOrders()) {
      degrees.push_back(d);
    }

    return degrees;
  }

  /// @brief n_vertices
  /// @return
  virtual int NumberOfVertices() const {
    MIMI_FUNC()
    // return Mesh()->GetNV(); // <-this is only n corner vertices.
    return Mesh()->GetNodes()->Size() / MeshDim();
  }

  /// @brief n_elem
  /// @return
  virtual int NumberOfElements() const {
    MIMI_FUNC()
    return Mesh()->GetNE();
  }

  /// @brief n_b_elem
  /// @return
  virtual int NumberOfBoundaryElements() const {
    MIMI_FUNC()
    return Mesh()->GetNBE();
  }

  /// @brief n_sub_elem
  /// @return
  virtual int NumberOfSubelements() const {
    MIMI_FUNC()
    return Mesh()->GetNumFaces();
  }

  /// @brief elevates degrees. can set max_degrees for upper bound.
  /// @param degrees relative degrees to elevate
  /// @param max_degrees upper bound
  virtual void ElevateDegrees(const int degrees, const int max_degrees = 50) {
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

  virtual void Subdivide(const int n_subdivision) {
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

  /// @brief Sets boundary condition
  /// @param boundary_conditions
  virtual void
  SetBoundaryConditions(const std::shared_ptr<mimi::utils::BoundaryConditions>&
                            boundary_conditions) {
    MIMI_FUNC()

    boundary_conditions_ = boundary_conditions;
  }

  virtual std::shared_ptr<mimi::utils::BoundaryConditions>
  GetBoundaryConditions() {
    MIMI_FUNC()
    return boundary_conditions_;
  }

  /// @brief finds true dof ids for each boundary this also finds zero_dofs_
  virtual void FindBoundaryDofIds() {
    MIMI_FUNC()

    // find all true dof ids
    for (auto& [key, fes] : fe_spaces_) {
      mimi::utils::PrintDebug("Finding boundary dofs for", key, "FE Space.");

      const int max_bdr_id = fes.fe_space_->GetMesh()->bdr_attributes.Max();

      // loop each bdr.
      for (int i{}; i < max_bdr_id; ++i) {
        // fespace's dim
        for (int j{}; j < fes.fe_space_->GetVDim(); ++j) {
          // mark only bdr id for this loop
          mfem::Array<int> bdr_id_query(max_bdr_id);
          bdr_id_query = 0;    // clear
          bdr_id_query[i] = 1; // mark

          // query
          fes.fe_space_->GetEssentialTrueDofs(bdr_id_query,
                                              fes.boundary_dof_ids_[i][j],
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
          fes.zero_dofs_.Append(fes.boundary_dof_ids_[bid][dim]);
        }
      }
      // on second thought, it is a bit harmless.
      fes.zero_dofs_.Sort();
      fes.zero_dofs_.Unique();
    }
  }

  virtual py::dict GetNurbs() {
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

  virtual void AddSpline(std::string const& s_name,
                         std::shared_ptr<splinepy::py::PySpline> spline) {
    MIMI_FUNC()

    splines_[s_name] = spline;
    mimi::utils::PrintInfo("I got a spline", spline->WhatAmI());
  }

  /// Setup multi threading basics. currently relevant only for OMP
  virtual void SetupNTheads(const int n_threads) {
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

  virtual void Setup(const int nthreads = -1) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("Derived class need to implement Setup().");
  };

  /// as newton solvers are separetly by name, you can configure newton solvers
  /// by name
  virtual void ConfigureNewton(const std::string& name,
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

  /// get final norms. can be used for augmented langrange iterations
  virtual py::tuple NewtonFinalNorms(const std::string& name) const {
    MIMI_FUNC()

    auto& newton = newton_solvers_.at(name);
    return py::make_tuple(newton->GetFinalRelNorm(), newton->GetFinalNorm());
  }

  virtual void UpdateContactLagrange() {
    MIMI_FUNC()

    PrepareALM();
    ALM.UpdateContactLagrange();
  }

  virtual void FillContactLagrange(const double value) {
    MIMI_FUNC()

    PrepareALM();
    ALM.FillContactLagrange(value);
  }

  /// @brief sets second order system with given ptr and takes ownership.
  /// @param oper2
  /// @param ode2
  virtual void SetDynamicSystem2(mfem::SecondOrderTimeDependentOperator* oper2,
                                 mimi::solvers::OdeBase* ode2) {

    MIMI_FUNC()

    oper2_ = std::unique_ptr<mfem::SecondOrderTimeDependentOperator>(oper2);
    ode2_solver_ = std::unique_ptr<mimi::solvers::OdeBase>(ode2);

    // ode solvers also wants to know dirichlet dofs
    auto* op_base = dynamic_cast<mimi::operators::OperatorBase*>(oper2_.get());
    ode2_solver_->SetupDirichletDofs(op_base->dirichlet_dofs_);
  }

  virtual double CurrentTime() const { MIMI_FUNC() return t_; }

  virtual double GetTimeStepSize() const { MIMI_FUNC() return dt_; }

  virtual void SetTimeStepSize(const double dt) {
    MIMI_FUNC()

    dt_ = dt;
  }

  virtual py::array_t<double> LinearFormView2(const std::string lf_name) {
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

  virtual py::array_t<double> SolutionView(const std::string& fes_name,
                                           const std::string& component_name) {
    MIMI_FUNC()

    auto& fes = fe_spaces_.at(fes_name);
    auto& grid_func = fes.grid_functions_.at(component_name);

    return mimi::py::NumpyView<double>(grid_func,
                                       grid_func.Size()); // will be raveled.
  }

  virtual py::array_t<int>
  BoundaryDofIds(const std::string& fes_name, const int& bid, const int& dim) {
    MIMI_FUNC()

    auto& fes = fe_spaces_.at(fes_name);
    auto& mfem_array = fes.boundary_dof_ids_.at(bid).at(dim);

    return mimi::py::NumpyCopy<int>(mfem_array, mfem_array.Size());
  }

  virtual py::array_t<int> ZeroDofIds(const std::string& fes_name) {
    MIMI_FUNC()

    auto& fes = fe_spaces_.at(fes_name);

    return mimi::py::NumpyCopy<int>(fes.zero_dofs_, fes.zero_dofs_.Size());
  }

  virtual std::shared_ptr<mimi::forms::Nonlinear>
  NonlinearForm2(const std::string& nlf_name) {
    MIMI_FUNC()

    assert(oper2_);

    auto* mimi_oper2 =
        dynamic_cast<mimi::operators::OperatorBase*>(oper2_.get());

    if (!mimi_oper2) {
      mimi::utils::PrintAndThrowError(
          "2nd order dynamic system does not exist yet.");
    }

    return mimi_oper2->nonlinear_forms_.at(nlf_name);
  }

  virtual void StepTime2() {
    MIMI_FUNC()

    assert(x2_);
    assert(x2_dot_);

    ode2_solver_->StepTime2(*x2_, *x2_dot_, t_, dt_);
  }

  virtual void FixedPointSolve2() {
    MIMI_FUNC()

    assert(x2_);
    assert(x2_dot_);

    ode2_solver_->FixedPointSolve2(*x2_, *x2_dot_, t_, dt_);
  }

  virtual void FixedPointAdvance2(mfem::Vector& fp_x, mfem::Vector& fp_v) {
    MIMI_FUNC()

    fp_x.SetSize(x2_->Size());
    fp_v.SetSize(x2_->Size());

    ode2_solver_->FixedPointAdvance2(fp_x, fp_v, t_, dt_);
  }

  // python returning version of fixed point advanced
  virtual py::tuple FixedPointAdvance2() {
    MIMI_FUNC()

    FixedPointAdvance2(fixed_point_advanced_x_, fixed_point_advanced_v_);

    return py::make_tuple(
        mimi::utils::NumpyView<double>(fixed_point_advanced_x_,
                                       fixed_point_advanced_x_.Size()),
        mimi::utils::NumpyView<double>(fixed_point_advanced_v_,
                                       fixed_point_advanced_v_.Size()));
  }

  virtual void PrepareALM() {
    MIMI_FUNC()

    if (ALM.contact_form && ALM.newton_solver) {
      return;
    }
    auto* mimi_oper =
        dynamic_cast<mimi::operators::OperatorBase*>(oper2_.get());
    if (mimi_oper) {
      ALM.contact_form = mimi_oper->nonlinear_forms_.at("contact");
    } else {
      mimi::utils::PrintAndThrowError(
          "Failed to cast operator to mimi's base.");
    }
    // if there's more than one newton, exit.
    // otherwise, we need to pass the name of the newton
    if (newton_solvers_.size() != 1) {
      mimi::utils::PrintAndThrowError(
          "There are more than one newton solvers. Please extend "
          "PySolid::PrepareALM to accept solver key.");
    }
    // easy way to unpack one elem
    ALM.newton_solver = newton_solvers_.begin()->second;
  }

  virtual void FixedPointALMSolve2(int n_outer,
                                   const int n_inner,
                                   const int n_final,
                                   const double rel_tol,
                                   const double abs_tol,
                                   const double gap_tol) {
    MIMI_FUNC()

    PrepareALM();

    const bool prev_itermode = ALM.newton_solver->iterative_mode;
    const bool

        // first iteration
        ALM.newton_solver->SetRelTol(rel_tol);
    ALM.newton_solver->SetAbsTol(abs_tol);
    ALM.newton_solver->SetMaxIter(n_inner);
    ALM.newton_solver->iterative_mode = false;
    FixedPointSolve2();
    FixedPointAdvance2();
    --n_outer;
    double gap = ALM.GapNorm(fixed_point_advanced_x_);

    auto is_converged = [&]() -> bool {
      if (ALM.Converged(gap, gap_tol)) {
        mimi::utils::PrintInfo("FixedPointALMSolve2 successful. Gap:", gap);
        ALM.newton_solver->iterative_mode = prev_itermode;
        return true;
      }
      return false;
    };

    if (is_converged())
      return;
    ALM.UpdateContactLagrange();

    // do the rest loops
    ALM.newton_solver->iterative_mode = true;
    for (int i{}; i < n_outer; ++i) {
      FixedPointSolve2();
      FixedPointAdvance2();
      gap = ALM.GapNorm(fixed_point_advanced_x_);
      if (is_converged())
        return;
      ALM.UpdateContactLagrange();
    }

    // last run - reducing coefficient
    ALM.newton_solver->SetMaxIter(n_final);
    FixedPointSolve2();
    FixedPointAdvance2();
    gap = ALM.GapNorm(fixed_point_advanced_x_);
  }

  virtual void AdvanceTime2() {
    MIMI_FUNC()

    assert(x2_);
    assert(x2_dot_);

    ode2_solver_->AdvanceTime2(*x2_, *x2_dot_, t_, dt_);
  }
};

} // namespace mimi::py
