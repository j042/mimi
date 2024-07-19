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
    std::string name{"None"};
    std::unique_ptr<mfem::FiniteElementSpace> fe_space{nullptr};
    std::map<int, std::map<int, mfem::Array<int>>> boundary_dof_ids;
    std::unordered_map<std::string, mfem::GridFunction> grid_functions;
    mfem::Array<int> zero_dofs;
    std::shared_ptr<mimi::utils::PrecomputedData> precomputed;
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

  // runtime comm
  std::shared_ptr<mimi::utils::RuntimeCommunication> runtime_communication_;

  /// @brief sets mesh
  /// @param fname
  virtual void ReadMesh(const std::string fname);

  virtual void SaveMesh(const std::string fname) const;

  /// @brief returns mesh. If it's missing, it will raise.
  /// @return
  virtual std::unique_ptr<mfem::Mesh>& Mesh();

  /// @brief returns mesh. If it's missing, it will raise.
  /// @return
  virtual const std::unique_ptr<mfem::Mesh>& Mesh() const;

  /// @brief returns mesh dim (geometry dim)
  /// @return
  virtual int MeshDim() const {
    MIMI_FUNC()

    return Mesh()->Dimension();
  }

  /// @brief degrees of mesh.
  /// @return std::vector<int>, but will be casted fo py::list
  virtual std::vector<int> MeshDegrees() const;

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

  virtual std::shared_ptr<mimi::utils::RuntimeCommunication>
  RuntimeCommunication();

  /// @brief elevates degrees. can set max_degrees for upper bound.
  /// @param degrees relative degrees to elevate
  /// @param max_degrees upper bound
  virtual void ElevateDegrees(const int degrees, const int max_degrees = 50);

  virtual void Subdivide(const int n_subdivision);

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

  /// @brief finds true dof ids for each boundary this also finds zero_dofs
  virtual void FindBoundaryDofIds();

  virtual py::dict GetNurbs();

  virtual void AddSpline(std::string const& s_name,
                         std::shared_ptr<splinepy::py::PySpline> spline);

  /// Setup multi threading basics. currently relevant only for OMP
  virtual void SetupNTheads(const int n_threads);

  virtual void Setup(const int nthreads = -1) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("Derived class need to implement Setup().");
  };

  virtual py::array_t<int> DofMap(const std::string& key) const;

  /// as newton solvers are separetly by name, you can configure newton solvers
  /// by name
  virtual void ConfigureNewton(const std::string& name,
                               const double rel_tol,
                               const double abs_tol,
                               const double max_iter,
                               const bool iterative_mode);

  /// get final norms. can be used for augmented langrange iterations
  virtual py::tuple NewtonFinalNorms(const std::string& name) const {
    MIMI_FUNC()

    auto& newton = newton_solvers_.at(name);
    return py::make_tuple(newton->GetFinalRelNorm(), newton->GetFinalNorm());
  }

  /// @brief sets second order system with given ptr and takes ownership.
  /// @param oper2
  /// @param ode2
  virtual void SetDynamicSystem2(mfem::SecondOrderTimeDependentOperator* oper2,
                                 mimi::solvers::OdeBase* ode2);

  virtual double CurrentTime() const { MIMI_FUNC() return t_; }

  virtual double GetTimeStepSize() const { MIMI_FUNC() return dt_; }

  virtual void SetTimeStepSize(const double dt) {
    MIMI_FUNC()

    dt_ = dt;
  }

  virtual py::array_t<double> LinearFormView2(const std::string lf_name);

  virtual py::array_t<double> SolutionView(const std::string& fes_name,
                                           const std::string& component_name);

  virtual py::array_t<int>
  BoundaryDofIds(const std::string& fes_name, const int& bid, const int& dim);

  virtual py::array_t<int> ZeroDofIds(const std::string& fes_name);

  virtual std::shared_ptr<mimi::forms::Nonlinear>
  NonlinearForm2(const std::string& nlf_name);

  virtual void StepTime2();

  virtual void FixedPointSolve2();

  virtual void FixedPointAdvance2(mfem::Vector& fp_x, mfem::Vector& fp_v);

  // python returning version of fixed point advanced
  virtual py::tuple FixedPointAdvance2();

  virtual py::tuple FixedPointAdvancedVectorViews();

  virtual void AdvanceTime2();
};

} // namespace mimi::py
