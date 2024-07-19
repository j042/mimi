#pragma once

#include "mimi/utils/print.hpp"

namespace mimi::materials {

// Non-Newtonian, shear thinning material (e.g. polymer melt)
class ShearThinningBehavior {
public:
  double cp_; // specific heat capacity
  double density_;
};

// Carreau model with Williams-Landel-Ferry (WLF) temperature correction
class CarreauWLF : public ShearThinningBehavior {
public:
  double a_t_; // WLF temperature correction factor
  double CarreauA_;
  double CarreauB_;
  double CarreauC_;
};

class FluidMaterialBase {
protected:
  std::shared_ptr<ShearThinningBehavior> shear_thinning_behavior_;

public:
  double density_;
  double viscosity_; // dynamic

  FluidMaterialBase() = default;

  /// @brief base line viscosity
  /// @return
  virtual double Viscosity() const {
    MIMI_FUNC()

    return viscosity_;
  }

  /// @brief temperature dependent viscosity
  /// @param temp
  /// @return
  virtual double Viscosity(const double temp) const {
    mimi::utils::PrintAndThrowError(
        "FluidMaterialBase - Temperature dependent viscosity not implemented");
  }

  virtual double Density() const {
    MIMI_FUNC()
    return density_;
  }

  virtual double Density() const {
    mimi::utils::PrintAndThrowError(
        "FluidMaterialBase - Temperature dependent density not implemented");
  }
};

class ShearThinning : public FluidMaterialBase {
public:
  double Viscosity(const double temp) const { return {}; }
}

} // namespace mimi::materials
