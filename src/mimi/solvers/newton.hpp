#pragma once

#include <tuple>

#include <mfem.hpp>

#include "mimi/integrators/materials.hpp"
#include "mimi/utils/ad.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::solvers {

/// @brief alias to mfem's newton solver
using Newton = mfem::NewtonSolver;

class LineSearchNewton : public Newton {
public:
  using Base_ = Newton;

  using Base_::Base_;

  /// @brief implements scaling factor usine line search adapted from
  /// https://github.com/LLNL/ExaConstit/blob/exaconstit-dev/src/mechanics_solver.cpp
  /// @param x
  /// @param b
  /// @return
  virtual void Mult(const mfem::Vector& b, mfem::Vector& x) const {

    MIMI_FUNC()

    int it;
    double norm0, norm, norm_goal;
    const bool have_b = (b.Size() == Height());

    // check iterative mode
    if (!Base_::iterative_mode) {
      x = 0.0;
    }

    // get residual
    Base_::oper->Mult(x, Base_::r);
    if (have_b) {
      r -= b;
    }

    // get initial norm
    norm0 = norm = Base_::initial_norm = Norm(Base_::r);
    if (Base_::print_options.first_and_last
        && !Base_::print_options.iterations) {
      mfem::out << "Newton iteration " << std::setw(2) << 0
                << " : ||r|| = " << norm << "...\n";
    }
    norm_goal = std::max(Base_::rel_tol * norm, Base_::abs_tol);

    Base_::prec->iterative_mode = false;
    double scale = 1.0;

    // x_{i+1} = x_i - [DF(x_i)]^{-1} [F(x_i)-b]
    for (it = 0; true; ++it) {

      // res norm checks
      MFEM_ASSERT(mfem::IsFinite(norm), "norm = " << norm);
      if (Base_::print_options.iterations) {
        mfem::out << "Newton iteration " << std::setw(2) << it
                  << " : ||r|| = " << norm;
        if (it > 0) {
          mfem::out << ", ||r||/||r_0|| = " << norm / norm0;
        }
        mfem::out << '\n';
      }
      Base_::Monitor(it, norm, Base_::r, x);

      // exit? - norm check
      if (norm <= norm_goal) {
        Base_::converged = true;
        break;
      }

      // exit? - n iter check
      if (it >= max_iter) {
        Base_::converged = false;
        break;
      }

      // set dr/du
      Base_::grad = &Base_::oper->GetGradient(x);
      Base_::prec->SetOperator(*Base_::grad);

      if (Base_::lin_rtol_type) {
        AdaptiveLinRtolPreSolve(x, it, norm);
      }

      // solve linear system
      Base_::prec->Mult(Base_::r, c); // c = [DF(x_i)]^{-1} [F(x_i)-b]

      if (Base_::lin_rtol_type) {
        AdaptiveLinRtolPostSolve(c, Base_::r, it, norm);
      }

      // now, line search
      // turn off the (plastic) state accumulation
      mimi::integrators::MaterialState::freeze_ = true;
      mfem::Vector tmp_x(x); // copy init

      // full step
      add(tmp_x, -1.0, Base_::c, tmp_x);
      oper->Mult(tmp_x, Base_::r);
      if (have_b) {
        r -= b;
      }

      const double q1 = norm; // norm from previous step
      const double q3 = Base_::Norm(Base_::r);

      // half step
      tmp_x = x;
      add(tmp_x, -0.5, Base_::c, tmp_x);
      Base_::oper->Mult(tmp_x, Base_::r);
      if (have_b) {
        Base_::r -= b;
      }
      const double q2 = Base_::Norm(Base_::r);

      // line search mult finished - unfreeze
      mimi::integrators::MaterialState::freeze_ = false;

      const double eps =
          (3.0 * q1 - 4.0 * q2 + q3) / (4.0 * (q1 - 2.0 * q2 + q3));
      if ((q1 - 2.0 * q2 + q3) > 0 && eps > 0 && eps < 1) {
        scale = eps;
      } else if (q3 < q1) {
        scale = 1.0;
      } else {
        scale = 0.05;
      }

      if (Base_::print_options.iterations) {
        mimi::utils::PrintInfo("The relaxation factor for this iteration is",
                               scale);
      }

      // scale too small?
      if (std::abs(scale) < 1e-12) {
        Base_::converged = 0;
        break;
      }

      // update solution
      add(x, -scale, Base_::c, x);

      // get current residual
      Base_::oper->Mult(x, Base_::r);
      if (have_b) {
        Base_::r -= b;
      }

      // current norm
      norm = Base_::Norm(Base_::r);
    }

    Base_::final_iter = it;
    Base_::final_norm = norm;

    if (Base_::print_options.summary
        || (!Base_::converged && Base_::print_options.warnings)
        || Base_::print_options.first_and_last) {
      mfem::out << "Newton: Number of iterations: " << final_iter << '\n'
                << "   ||r|| = " << final_norm << '\n';
    }
    if (!Base_::converged
        && (Base_::print_options.summary || Base_::print_options.warnings)) {
      mfem::out << "Newton: No convergence!\n";
    }
  }
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
  double fl = f(lower_bound, params...);
  double fh = f(upper_bound, params...);

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
        mimi::utils::PrintWarning(
            "ScalarSolve: failed to converge in allotted iterations.");
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
