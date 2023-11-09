#pragma once

#include <mfem.hpp>

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

} // namespace mimi::solvers
