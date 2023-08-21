#pragma once

#include <map>
#include <memory>
#include <string>
#include <unordered_map>

/* pybind11 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// mimi
#include "mimi/utils/boundary_conditions.hpp"
#include "mimi/solvers/ode.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::py {

class PySolid {
protected:
  // second order time dependent systems
  std::unique_ptr<mimi::solvers::OdeBase> ode2_solver_ = nullptr;
  std::unique_ptr<mfem::SecondOrderTimeDependentOperator> oper2_ = nullptr;

  // first order time dependent systems
  std::unique_ptr<mimi::solvers::OdeBase> ode1_solver_ = nullptr;
  std::unique_ptr<mfem::TimeDependentOperator> oper1_ = nullptr;

  // mesh
  std::unique_ptr<mfem::Mesh> mesh_ = nullptr;

  struct FESpace {
    std::string name_{"None"};
    std::unique_ptr<mfem::FiniteElementSpace> fe_space_{nullptr};
    std::map<int, std::map<int, mfem::Array<int>>> boundary_dof_ids_;
    std::unordered_map<std::string, mfem::GridFunction> grid_functions_;
    mfem::Array<int> zero_dofs_;
  };

  // there can be multiple fe spaces
  std::unordered_map<std::string, FESpace> fe_spaces_{};

  // bc manager
  std::shared_ptr<mimi::utils::BoundaryCondition> boundary_conditions_ =
      nullptr;

  // fsi coupling
  // a raw rhs vector for fsi loads
  std::shared_ptr<mfem::Vector> rhs_vector_ = nullptr;
  mfem::Vector intermediate_x_;

  // holder for coefficients
  std::map<std::string, std::shared_ptr<mfem::Coefficients>> coefficients_;

  // holder for vector coefficients
  std::map<std::string, std::shared_ptr<mfem::VectorCoefficients>>
      vector_coefficients_;

  // current time, CET
  double t_{0.0};

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
      mimi::utils::PrintD("read NURBS mesh");
    } else {
      mimi::utils::PrintE(fname, "Does not contain NURBS mesh.");
    }
  }

  /// @brief returns mesh. If it's missing, it will raise.
  /// @return
  virtual auto& Mesh() {
    MIMI_FUNC()

    if (!mesh_) {
      mimi::utils::PrintE("Mesh not set.");
    }

    return mesh_;
  }

  /// @brief returns mesh. If it's missing, it will raise.
  /// @return
  virtual const auto& Mesh() const {
    MIMI_FUNC()

    if (!mesh_) {
      mimi::utils::PrintE("Mesh not set.");
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
    return Mesh()->GetNV();
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

  virtual void Subdivide(const int n_subdivision) {
    MIMI_FUNC()

    mimi::utils::PrintD("n_subdivision:", n_subdivision);
    mimi::utils::PrintD("Number of elements before subdivision: ",
                        NumberOfElements());

    if (n_subdivision > 0) {
      for (i{}; i < n_subdivision; ++i) {
        Mesh()->UniformRefinements();
      }
    }

    mimi::utils::PrintD("Number of elements after subdivision: ",
                        NumberOfElements());
  }

  /// @brief Sets boundary condition
  /// @param boundary_conditions
  virtual void
  SetBoundaryCondition(const std::shared_ptr<mimi::utils::BoundaryCondition>&
                           boundary_conditions) {
    MIMI_FUNC()

    boundary_condition_ = boundary_condition;
  }

  virtual std::shared_ptr<mimi::utils::BoundaryCondition>
  GetBoundaryCondition() {
    MIMI_FUNC()
    return boundary_condition_;
  }

  /// @brief finds true dof ids for each boundary this also finds zero_dofs_
  virtual void FindBoundaryDofIds() {
    MIMI_FUNC()

    // find all true dof ids
    for (auto const& [key, fes] : *fe_spaces_) {
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
    for (auto const& [name, fes] : *fe_spaces_) {
      for (auto const& [bid, dim] :
           boundary_conditions_->InitialConfiguration().Dirichlet()) {

        mimi::utils::PrintDebug(
            "For FE Space",
            name,
            "- finding boundary dofs for initial configuration dirichlet bcs.",
            "bid:",
            bid,
            "dim:",
            dim);

        // append saved dofs
        // may have duplicating dofs, harmless.
        fes.zero_dofs_.Append(fes.boundary_dof_ids_[bid][dim]);
      }
    }
  }

  virtual void Setup() {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("Derived class need to implement Setup().");
  };

  /// @brief sets second order system with given ptr and takes ownership.
  /// @param oper2
  /// @param ode2
  virtual void SetDynamicSystem2(mfem::SecondOrderTimeDependentOperator* oper2,
                                 mimi::solvers::OdeBase* ode2) {

    MIMI_FUNC()

    oper2_ = std::unique_ptr<mfem::SecondOrderTimeDependentOperator>(oper2);
    ode2_ = std::unique_ptr<mimi::solvers::OdeBase>(ode2);
  }
};

} // namespace mimi::py
