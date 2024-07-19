#pragma once

#include <map>
#include <memory>
#include <set>

#include <mfem.hpp>

// mimi
#include "mimi/coefficients/nearest_distance.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::utils {

/// @brief boundary condition marker
class BCMarker {
public:
  bool initial_config_;
  std::map<int, std::set<int>> dirichlet_;
  std::map<int, double> pressure_;
  std::map<int, std::map<int, double>> traction_;
  std::map<int, double> body_force_;
  std::map<int, std::shared_ptr<mimi::coefficients::NearestDistanceBase>>
      contact_;
  std::map<int, std::map<int, double>> constant_velocity_;
  std::map<int, int> periodic_boundaries_;

  void OnlyForInitialConfig(const std::string& b_name);

  void OnlyForCurrentConfig(const std::string& b_name);

  /// @brief dirichlet bc, currently only applies 0.
  /// @param bid
  /// @param dim
  /// @return
  BCMarker& Dirichlet(const int bid, const int dim);
  BCMarker& PrintDirichlet();

  /// @brief pressure bc
  /// @param bid
  /// @param value
  /// @return
  BCMarker& Pressure(const int bid, const double value);
  BCMarker& PrintPressure();

  /// @brief traction bc
  /// @param bid
  /// @param dim
  /// @param value
  /// @return
  BCMarker& Traction(const int bid, const int dim, const double value);
  BCMarker& PrintTraction();

  /// @brief body force (per unit volume). Technically not a `B`C.
  /// @param bid
  /// @param dim
  /// @param value
  /// @return
  BCMarker& BodyForce(const int dim, const double value);
  BCMarker& PrintBodyForce();

  BCMarker&
  Contact(const int bid,
          const std::shared_ptr<mimi::coefficients::NearestDistanceBase>&
              nearest_distance_coeff);
  BCMarker& PrintContact();

  BCMarker& ConstantVelocity(const int bid, const int dim, const double value);
  BCMarker& PrintConstantVelocity();

  /// values to be used in NURBSExtension::ConnectBoundaries(b0, b1)
  /// those are "fortran" numbering unlike other BC
  BCMarker& PeriodicBoundary(const int bid0, const int bid1);
  BCMarker& PrintPeriodicBoundary();
};

/// @brief class to hold boundary conditions
class BoundaryConditions {
public:
  BCMarker initial_;
  BCMarker current_;

  BCMarker& InitialConfiguration();
  BCMarker& CurrentConfiguration();

  /// @brief BC config print
  void Print();
};

/// @brief minimal type to apply dynamic bc. now we only have constant velocity
class TimeDependentDirichletBoundaryCondition {
public:
  std::map<int, std::map<int, mfem::Array<int>>>* boundary_dof_ids_;
  BoundaryConditions* dynamic_bc_;

  Vector<double> saved_x_;
  Vector<double> saved_v_;
  Vector<double> saved_a_;

  /// Checks if we have pointers to boundary_dof_ids_ and dynamic_bc_
  bool HasDynamicDirichlet();

  /// @brief apply bounday conditions. note that those are alpha values.
  /// @param t
  /// @param dt
  /// @param x
  /// @param v
  /// @param a
  void Apply(const double t,
             const double dt,
             const mfem::Vector& x,
             const mfem::Vector& v,
             const mfem::Vector& a,
             mfem::Vector& xa,
             mfem::Vector& va,
             mfem::Vector& aa);

  /// @brief restores applied BC data from Apply(). It is for g
  /// @param x
  /// @param v
  /// @param a
  void Restore(mfem::Vector& x, mfem::Vector& v, mfem::Vector& a) const;
};

} // namespace mimi::utils
