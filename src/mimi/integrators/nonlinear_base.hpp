#pragma once

#include <cmath>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/utils/precomputed.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

class NonlinearBase : public mfem::NonlinearFormIntegrator {
protected:
  std::shared_ptr<mimi::utils::PrecomputedData> precomputed_;
  std::unique_ptr<mimi::utils::Data<mfem::Vector>> element_vectors_;
  std::unique_ptr<mimi::utils::Data<mfem::DenseMatrix>> element_matrices_;

public:
  std::string name_;

  /// basic ctor saved ptr to precomputed and name to use as key in precomputed
  NonlinearBase(
      const std::string& name,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : name_(name),
        precomputed_(precomputed) {}

  ///
  virtual const std::string& Name() const { return name_; }

  virtual void AssembleFaceResidual(const mfem::Vector& current_x) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("AssembleFaceResidual not implemented");
  }

  virtual void AssembleFaceResidualGrad(const mfem::Vector& current_x) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("AssembleFaceResidualGrad not implemented");
  }

  /// implemement criterium in case norm(residual) == 0 is not enough
  /// default is true;
  virtual bool Satisfied() const {
    MIMI_FUNC()

    return true;
  }
};

} // namespace mimi::integrators
