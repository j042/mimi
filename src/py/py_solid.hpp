#include <memory>

/* pybind11 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// splinepy
#include <splinepy/py/py_spline.hpp>

// mimi
#include "mimi/utils/print.hpp"

namespace mimi::py {
class PySolid {
protected:
  std::unique_ptr<mimi::SecondOrderODESolver> ode2_solver_ = nullptr;
  std::unique_ptr<mimi::NonLinearTimeDependentOperator> operator_ = nullptr;
  std::unique_ptr<mimi::FirstOrderODESolver> ode1_solver_ = nullptr;
  std::unique_ptr<mfem::Mesh> mesh_ = nullptr;
  std::unique_ptr<mfem::FiniteElementSpace> fe_space_ = nullptr;
  std::unique_ptr<mfem::LinearForm> rhs_linear_form_ = nullptr;

  // bc manager
  std::map<int, std::map<int, bool>> dirichlet_bcs_;
  std::map<int, std::map<int, double>> pressure_bcs_;
  std::map<int, std::map<int, double>> traction_bcs_;
  std::map<int, double> body_forces_;
  std::map<int std::map<int, mfem::Array<int>>> boundary_dof_ids_;
  mfem::Array<int> ess_tdof_list_;

  // fsi coupling
  mfem::Vector rhs_forces_;
  mfem::Vector intermediate_x_;
  mfem::Vector intermediate_displacement_;
  mfem::Vector w_; // internal vector for computing

  // current state
  double t_{0.0};

public:

  PySolid() = default;

  /// @brief sets mesh
  /// @param fname 
  void SetMesh(const std::string fname) {
    MIMI_FUNC()

    const char* fname_char = fname.c_str();

    // read mesh and save
    mesh_ = std::make_unique<mfem::Mesh>(fname_char, 1, 1);

    // nurbs check
    if (mesh_->NURBSext) {
        mimi::utils::PrintD("read NURBS mesh");
    } else {
        mimi::utils::PrintE(fname, "Does not contain NURBS mesh.");
    }

  }

  /// @brief returns mesh. If it's missing, it will raise.
  /// @return 
  constexpr auto& Mesh() {
    MIMI_FUNC()

    if (!mesh_) {
        mimi::utils::PrintE("Mesh not set.");
    }

    return mesh_;
  }

  /// @brief returns mesh. If it's missing, it will raise.
  /// @return 
  constexpr const auto& Mesh() const {
    MIMI_FUNC()

    if (!mesh_) {
        mimi::utils::PrintE("Mesh not set.");
    }

    return mesh_;
  }

  /// @brief returns mesh dim (geometry dim)
  /// @return 
  int MeshDim() const {
    MIMI_FUNC()

    return Mesh()->Dimension();
  }

  /// @brief degrees of mesh. 
  /// @return std::vector<int>, but will be casted fo py::list
  std::vector<int> MeshDegrees() const {
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
  int NumberOfVertices() const {
    return Mesh()->GetNV();
  }

  /// @brief n_elem
  /// @return 
  int NumberOfElements() const {
    return Mesh()->GetNE();
  }

  /// @brief n_b_elem
  /// @return 
  int NumberOfBoundaryElements() const {
    return Mesh()->GetNBE();
  }

  /// @brief n_sub_elem
  /// @return 
  int NumberOfSubelements() const {
    return Mesh()->GetNumFaces();
  }

  /// @brief elevates degrees. can set max_degrees for upper bound.
  /// @param degrees relative degrees to elevate
  /// @param max_degrees upper bound
  void ElevateDegrees(const int degrees, const int max_degrees=50) {
    MIMI_FUNC()

    mimi::utils::PrintD("degrees input:", degrees);
    mimi::utils::PrintD("max_degrees set to", max_degrees);

    if (degrees > 0) {
        Mesh()->DegreeElevate(degrees, max_degrees);
    } else {
        mimi::utils::PrintW(degrees, "is invalid input. Skipping.");
    }

    // FYI
    auto ds = MeshDegrees();
    mimi::utils::PrintD("current degrees:");
    for (int i{}; i < MeshDim(); ++i) {
        mimi::utils::PrintD("dim", i, ":", ds[i]);
    }
  }

  void Subdivide(const int n_subdivision) {
    MIMI_FUNC()

    mimi::utils::PrintD("n_subdivision:", n_subdivision);
    mimi::utils::PrintD("Number of elements before subdivision: ", NumberOfElements());

    if (n_subdivision > 0) {
        for (i{}; i < n_subdivision; ++i) {
            Mesh()->UniformRefinements();
        }
    }

    mimi::utils::PrintD("Number of elements after subdivision: ", NumberOfElements());

  }

  void AddDirichletBC(int boundary_id, int dof_id) {
    MIMI_FUNC()

    dirichlet_bcs_[boundary_id][dof_id] = true;
  }

  void AddPressureBC(
    const int boundary_id,
    const int dof_id,
    const double value,
    const on_reference=true) {
    MIMI_FUNC()

    pressure_bcs_[boundary_id][dof_id] = value;
  }

  void AddTractionBC(int boundary_id, int dof_id, double value) {
    MIMI_FUNC()

    traction_bcs_[boundary_id][dof_id] = value;
  }

  void AddConstantForce

};

}