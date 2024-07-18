#pragma once

namespace mimi::materials {

class FluidMaterialBase {
public:
  double viscosity_; // dynamic

  // Non-Newtonian, shear thinning material (e.g. polymer melt)
  struct ShearThinningBase {
    double cp_; // specific heat capacity
    double density_;
  };

  // Carreau model with Williams-Landel-Ferry (WLF) temperature correction
  struct CarreauWLF : public ShearThinningBase {
    double a_t_;      // WLF temperature correction factor
    double CarreauA_;
    double CarreauB_;
    double CarreauC_;
  };

  FluidMaterialBase() = default;
};
} // namespace mimi::materials
