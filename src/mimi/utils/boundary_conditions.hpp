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

/// @brief class to hold boundary conditions
class BoundaryConditions {
public:
  /// @brief boundary condition marker
  struct BCMarker {
    bool initial_config_;
    std::map<int, std::set<int>> dirichlet_;
    std::map<int, double> pressure_;
    std::map<int, std::map<int, double>> traction_;
    std::map<int, double> body_force_;
    std::map<int, std::shared_ptr<mimi::coefficients::NearestDistanceBase>>
        contact_;
    std::map<int, std::map<int, double>> constant_velocity_;

    void OnlyForInitialConfig(const std::string& b_name) {
      if (!initial_config_) {
        mimi::utils::PrintAndThrowError(b_name,
                                        "boundary condition is currently only "
                                        "available for initial config.");
      }
    }

    void OnlyForCurrentConfig(const std::string& b_name) {
      if (initial_config_) {
        mimi::utils::PrintAndThrowError(b_name,
                                        "boundary condition is currently only "
                                        "available for current config.");
      }
    }

    /// @brief dirichlet bc, currently only applies 0.
    /// @param bid
    /// @param dim
    /// @return
    BCMarker& Dirichlet(const int bid, const int dim) {
      MIMI_FUNC()

      OnlyForInitialConfig("Dirichlet");

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

      OnlyForInitialConfig("Pressure");

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

      OnlyForInitialConfig("Traction");

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

      OnlyForInitialConfig("BodyForce");

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

      OnlyForCurrentConfig("Contact");

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

    BCMarker&
    ConstantVelocity(const int bid, const int dim, const double value) {
      MIMI_FUNC()

      OnlyForInitialConfig("ConstantVelocity");
      Dirichlet(bid, dim); // need dirichlet on those boundaries automatically

      constant_velocity_[bid][dim] = value;

      return *this;
    }

    BCMarker& PrintConstantVelocity() {
      MIMI_FUNC()

      mimi::utils::PrintInfo("constant velocity bc (bid | dim | value):");
      for (auto const& [bid, dim_value] : constant_velocity_)
        for (auto const& [dim, value] : dim_value) {
          mimi::utils::PrintInfo("  ", bid, "|", dim, "|", value);
        }
      return *this;
    }
  };

  BCMarker initial_;
  BCMarker current_;

  BCMarker& InitialConfiguration() {
    MIMI_FUNC() initial_.initial_config_ = true;
    return initial_;
  }
  BCMarker& CurrentConfiguration() {
    MIMI_FUNC() initial_.initial_config_ = false;
    return current_;
  }

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
        .PrintConstantVelocity()
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

/// @brief minimal type to apply dynamic bc. now we only have constant velocity
struct TimeDependentDirichletBoundaryCondition {
  std::map<int, std::map<int, mfem::Array<int>>>* boundary_dof_ids_;
  BoundaryConditions* dynamic_bc_;

  Vector<double> saved_x_;
  Vector<double> saved_v_;
  Vector<double> saved_a_;

  bool HasDynamicDirichlet() const {
    MIMI_FUNC()
    return (boundary_dof_ids_ && dynamic_bc_);
  }

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
             mfem::Vector& aa) {
    MIMI_FUNC()
    if (HasDynamicDirichlet()) {
      mimi::utils::PrintInfo("Applying dynamic dirichlet");

      const auto& b_tdof = *boundary_dof_ids_;

      const double* x_d = x.GetData();
      const double* v_d = v.GetData();
      const double* a_d = a.GetData();
      double* xa_d = xa.GetData();
      double* va_d = va.GetData();
      double* aa_d = aa.GetData();

      const int size = x.Size();
      saved_x_.resize(size);
      saved_v_.resize(size);
      saved_a_.resize(size);

      // apply constant velocity - if we want direct dirichlet, we will of
      // course have to reconsider order of this function
      for (auto const& [bid, dim_value] :
           dynamic_bc_->initial_.constant_velocity_)
        for (auto const& [dim, value] : dim_value) {
          const mfem::Array<int>& tdof = b_tdof.at(bid).at(dim);
          // loop tdof
          for (const auto i : tdof) {
            aa_d[i] = 0.;
            va_d[i] = value;
            // x is converged solution from one step before
            xa_d[i] = x_d[i] + value * dt; // this is true dt

            // we save those values so that we can apply it at the end of the
            // stepping
            saved_x_[i] = xa_d[i];
            saved_v_[i] = va_d[i];
            saved_a_[i] = aa_d[i];
          }
        }
    } else {
      mimi::utils::PrintDebug(" NO dynamic dirichlet");
    }
  }

  void Restore(mfem::Vector& x, mfem::Vector& v, mfem::Vector& a) const {

    if (HasDynamicDirichlet()) {
      mimi::utils::PrintInfo("Restoring dynamic dirichlet");

      const auto& b_tdof = *boundary_dof_ids_;

      double* x_d = x.GetData();
      double* v_d = v.GetData();
      double* a_d = a.GetData();

      // apply constant velocity - if we want direct dirichlet, we will of
      // course have to reconsider order of this function
      for (auto const& [bid, dim_value] :
           dynamic_bc_->initial_.constant_velocity_)
        for (auto const& [dim, value] : dim_value) {
          const mfem::Array<int>& tdof = b_tdof.at(bid).at(dim);
          // loop tdof
          for (const auto i : tdof) {
            x_d[i] = saved_x_[i];
            v_d[i] = saved_v_[i];
            a_d[i] = saved_a_[i];
          }
        }
    } else {
      mimi::utils::PrintDebug("Restoring NO dynamic dirichlet");
    }
  }
};

} // namespace mimi::utils
