#pragma once

#include <map>
#include <memory>
#include <set>

// mimi
#include "mimi/coefficients/nearest_distance.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::utils {

/// @brief class to hold boundary conditions
class BoundaryConditions {
public:
  /// @brief boundary condition marker
  struct BCMarker {
    std::map<int, std::set<int>> dirichlet_;
    std::map<int, double> pressure_;
    std::map<int, std::map<int, double>> traction_;
    std::map<int, double> body_force_;
    std::map<int, std::shared_ptr<mimi::coefficients::NearestDistanceBase>>
        contact_;

    /// @brief dirichlet bc, currently only applies 0.
    /// @param bid
    /// @param dim
    /// @return
    BCMarker& Dirichlet(const int bid, const int dim) {
      MIMI_FUNC()

      dirichlet_[bid].insert(dim);
      return *this;
    }

    BCMarker& PrintDirichlet() {
      MIMI_FUNC()

      mimi::utils::PrintInfo("dirichlet bc (bid | dim):");
      for (auto const& [bid, dims] : dirichlet_) {
        for (auto const& dim : dims) {
          mimi::utils::PrintInfo("  ", bid, "|", dim);
        }
      }

      return *this;
    }

    /// @brief pressure bc
    /// @param bid
    /// @param value
    /// @return
    BCMarker& Pressure(const int bid, const double value) {
      MIMI_FUNC()

      pressure_[bid] = value;
      return *this;
    }

    BCMarker& PrintPressure() {
      MIMI_FUNC()

      mimi::utils::PrintInfo("pressure bc (bid | value):");
      for (auto const& [key, value] : pressure_) {
        mimi::utils::PrintInfo("  ", key, "|", value);
      }

      return *this;
    }

    /// @brief traction bc
    /// @param bid
    /// @param dim
    /// @param value
    /// @return
    BCMarker& Traction(const int bid, const int dim, const double value) {
      MIMI_FUNC()

      traction_[bid][dim] = value;
      return *this;
    }

    BCMarker& PrintTraction() {
      MIMI_FUNC()

      mimi::utils::PrintInfo("traction bc (bid | dim | value):");
      for (auto const& [bid, dim_value] : traction_)
        for (auto const& [dim, value] : dim_value) {
          mimi::utils::PrintInfo("  ", bid, "|", dim, "|", value);
        }

      return *this;
    }

    /// @brief body force (per unit volume). Technically not a `B`C.
    /// @param bid
    /// @param dim
    /// @param value
    /// @return
    BCMarker& BodyForce(const int dim, const double value) {
      MIMI_FUNC()

      body_force_[dim] = value;
      return *this;
    }

    BCMarker& PrintBodyForce() {
      MIMI_FUNC()

      mimi::utils::PrintInfo("body force (dim | value):");
      for (auto const& [dim, value] : body_force_)
        mimi::utils::PrintInfo("  ", dim, "|", value);

      return *this;
    }

    BCMarker&
    Contact(const int bid,
            const std::shared_ptr<mimi::coefficients::NearestDistanceBase>&
                nearest_distance_coeff) {
      MIMI_FUNC()

      contact_[bid] = nearest_distance_coeff;

      return *this;
    }

    BCMarker& PrintContact() {
      MIMI_FUNC()

      mimi::utils::PrintInfo("contact (bid | number of related objects):");
      for (auto const& [bid, nd_coeff] : contact_) {
        mimi::utils::PrintInfo("  ", bid, "|", nd_coeff->Size());
      }

      return *this;
    }
  };

  BCMarker initial_;
  BCMarker current_;

  BCMarker& InitialConfiguration() { MIMI_FUNC() return initial_; }
  BCMarker& CurrentConfiguration() { MIMI_FUNC() return current_; }

  /// @brief BC config print
  void Print() {
    MIMI_FUNC()
    mimi::utils::PrintInfo("************************************************");

    mimi::utils::PrintInfo("Boundary Condition Info");
    mimi::utils::PrintInfo("=======================");
    mimi::utils::PrintInfo("\nTo be applied on initial configuration:");
    InitialConfiguration()
        .PrintDirichlet()
        .PrintPressure()
        .PrintTraction()
        .PrintBodyForce()
        .PrintContact();
    mimi::utils::PrintInfo("\nTo be applied on current configuration:");
    CurrentConfiguration()
        .PrintDirichlet()
        .PrintPressure()
        .PrintTraction()
        .PrintBodyForce()
        .PrintContact();
    mimi::utils::PrintInfo("************************************************");
  }
};

} // namespace mimi::utils
