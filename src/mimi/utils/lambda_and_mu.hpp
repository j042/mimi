#pragma once

#include <utility>

namespace mimi::utils {
/// @brief Young's modulus and poisson's ratio to lambda and mu
/// @param young
/// @param poisson
/// @param lambda
/// @param mu
std::pair<double, double> ToLambdaAndMu(const double young,
                                        const double poisson) {
  return {young * poisson / ((1 + poisson) * (1 - 2 * poisson)), // lambda
          young / (2.0 * (1.0 + poisson))};                      // mu
}

} // namespace mimi::utils
