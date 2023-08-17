#pragma once

#include <mfem.hpp>

namespace mimi::solvers {

class NewtonLineSearch : public mfem::NewtonSolver {
public:
  using Base_ = mfem::NewtonSolver;

  /// @brief implements scaling factor usine line search adapted from
  /// https://github.com/LLNL/ExaConstit/blob/exaconstit-dev/src/mechanics_solver.cpp
  /// @param x
  /// @param b
  /// @return
  virtual double ComputeScalingFactor(const Vector& x, const Vector& b) const {

    // full step
    mfem::Vector tmp_x = x;
    mfem::add(tmp_x, -1.0, Base_::c, tmp_x);
    oper->Mult(tmp_x, Base_::r);
    if (have_b) {
      r -= b;
    }

    const double q1 = Base_::norm; // norm from previous step
    const double q3 = Norm(r);

    // half step
    tmp_x = x;
    add(tmp_x, -0.5, Base_::c, tmp_x);
    oper->Mult(x, Base_::r);
    if (Base_::have_b) {
      Base_::r -= b;
    }

    const double q2 = Norm(Base_::r);

    const double eps =
        (3.0 * q1 - 4.0 * q2 + q3) / (4.0 * (q1 - 2.0 * q2 + q3));
    double scale{0.05};
    if ((q1 - 2.0 * q2 + q3) > 0 && eps > 0 && eps < 1) {
      scale = eps;
    } else if (q3 < q1) {
      scale = 1.0;
    }

    if (print_level >= 0) {
      mfem::out << "The relaxation factor for this iteration is " << scale
                << "\n";
    }

    return scale;
  }
};

} // namespace mimi::solvers
