#pragma once

namespace mimi::materials {

class FluidMaterialBase {
public:
  double density_;
  double viscosity_; // dynamic

  FluidMaterialBase() = default;
};
} // namespace mimi::materials
