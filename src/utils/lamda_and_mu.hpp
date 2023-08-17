#pragma once

#include <utility>

/// @brief Young's modulus and poisson's ratio to lambda and mu
/// @param young
/// @param poisson
/// @param lambda
/// @param mu
std::pair<double, double> ToLambdaAndMu(const double young,
                                        const double poisson) {
  return {young / (3.0 * (1.0 - (2.0 * poisson))), // lambda
          young / (2.0 * (1.0 + poisson))};        // mu
}
