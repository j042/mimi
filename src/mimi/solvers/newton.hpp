#pragma once

#include <mfem.hpp>

#include "mimi/utils/ad.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::operators {
class NonlinearSolid;
}

namespace mimi::solvers {

/// @brief alias to mfem's newton solver
using Newton = mfem::NewtonSolver;

class LineSearchNewton : public Newton {
public:
  using Base_ = Newton;

  using Base_::Base_;

  /// pointer to nl oper to
  mimi::operators::NonlinearSolid* nl_oper_{nullptr};

  /// Mult is not thread safe, due to some mutable variables. so let's use this
  /// vector within Mult()
  mutable mfem::Vector line_search_temp_x_;
  mutable mfem::Vector best_x_;

  /// @brief implements scaling factor using line search adapted from
  /// https://github.com/LLNL/ExaConstit/blob/exaconstit-dev/src/mechanics_solver.cpp
  /// @param x
  /// @param b
  /// @return
  virtual void Mult(const mfem::Vector& b, mfem::Vector& x) const;
};

struct ScalarSolverOptions {
  double xtol;           ///< absolute tolerance on Newton correction
  double rtol;           ///< absolute tolerance on absolute value of residual
  unsigned int max_iter; ///< maximum allowed number of iterations
};

/// implementation taken from serac
/// thank you serac
/// ParamTypes should be just simple double
template<typename function, typename... ParamTypes>
auto ScalarSolve(function&& f,
                 double x0,
                 double lower_bound,
                 double upper_bound,
                 ScalarSolverOptions options,
                 ParamTypes... params) {

  using AD = mimi::utils::ADScalar<double, 1>;
  // param types should be a double.
  double x, df_dx;
  double fl = f(lower_bound, params...).GetValue();
  double fh = f(upper_bound, params...).GetValue();

  if (fl * fh > 0.) {
    mimi::utils::PrintAndThrowError(
        "ScalarSolve: root not bracketed by input bounds.");
  }

  unsigned int iterations = 0;
  bool converged = false;

  // handle corner cases where one of the brackets is the root
  if (fl == 0) {
    x = lower_bound;
    converged = true;
  } else if (fh == 0) {
    x = upper_bound;
    converged = true;
  }

  if (converged) {
    df_dx = f(AD(x, 0), params...).GetDerivatives(0);

  } else {
    // orient search so that f(xl) < 0
    double xl = lower_bound;
    double xh = upper_bound;
    if (fl > 0) {
      xl = upper_bound;
      xh = lower_bound;
    }

    // move initial guess if it is not between brackets
    if (x0 < lower_bound || x0 > upper_bound) {
      x0 = 0.5 * (lower_bound + upper_bound);
    }

    x = x0;
    double delta_x_old = std::abs(upper_bound - lower_bound);
    double delta_x = delta_x_old;
    auto R = f(AD(x, 0), params...);
    auto fval = R.GetValue();
    df_dx = R.GetDerivatives(0);

    while (!converged) {
      if (iterations == options.max_iter) {
        mimi::utils::PrintAndThrowError(
            "ScalarSolve: failed to converge in allotted iterations.",
            "delata_x",
            delta_x,
            "x_tol",
            options.xtol,
            "residual",
            fval,
            "r_tol",
            options.rtol);
        break;
      }

      // use bisection if Newton oversteps brackets or is not decreasing
      // sufficiently
      if ((x - xh) * df_dx - fval > 0 || (x - xl) * df_dx - fval < 0
          || std::abs(2. * fval) > std::abs(delta_x_old * df_dx)) {
        delta_x_old = delta_x;
        delta_x = 0.5 * (xh - xl);
        x = xl + delta_x;
        converged = (x == xl);
      } else { // use Newton step
        delta_x_old = delta_x;
        delta_x = fval / df_dx;
        auto temp = x;
        x -= delta_x;
        converged = (x == temp);
      }

      // function and jacobian evaluation
      R = f(AD(x, 0), params...);
      fval = R.GetValue();
      df_dx = R.GetDerivatives(0);

      // convergence check
      converged = converged || (std::abs(delta_x) < options.xtol)
                  || (std::abs(fval) < options.rtol);

      // maintain bracket on root
      if (fval < 0) {
        xl = x;
      } else {
        xh = x;
      }

      ++iterations;
    }
  }

  // Accumulate derivatives so that the user can get derivatives
  // with respect to parameters, subject to constraing that f(x, p) = 0 for all
  // p Conceptually, we're doing the following: [fval, df_dp] = f(get_value(x),
  // p) df = 0 for p in params:
  //   df += inner(df_dp, dp)
  // dx = -df / df_dx
  // constexpr bool contains_duals =
  //     (is_dual_number<ParamTypes>::value || ...) ||
  //     (is_tensor_of_dual_number<ParamTypes>::value || ...);
  // if constexpr (contains_duals) {
  //   auto [fval, df] = f(x, params...);
  //   auto         dx = -df / df_dx;
  //   SolverStatus status{.converged = converged, .iterations = iterations,
  //   .residual = fval}; return tuple{dual{x, dx}, status};
  // }
  // if constexpr (!contains_duals) {
  //   auto         fval = f(x, params...);
  //   SolverStatus status{.converged = converged, .iterations = iterations,
  //   .residual = fval}; return std::tuple{x, status};
  // }

  return x;
}

} // namespace mimi::solvers
