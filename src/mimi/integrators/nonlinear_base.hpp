#pragma once

#include <cmath>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/utils/precomputed.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

class NonlinearBase : public mfem::NonlinearFormIntegrator {
public:
  std::shared_ptr<mimi::utils::PrecomputedData> precomputed_;
  std::unique_ptr<mimi::utils::Data<mfem::Vector>> element_vectors_;
  std::unique_ptr<mimi::utils::Data<mfem::Vector>> boundary_element_vectors_;
  std::unique_ptr<mimi::utils::Data<mfem::DenseMatrix>> element_matrices_;
  std::unique_ptr<mimi::utils::Data<mfem::DenseMatrix>>
      boundary_element_matrices_;

  std::string name_;

  /// marker for this bdr face integ
  const mfem::Array<int>* boundary_marker_ = nullptr;

  /// ids of contributing boundary elements, extracted based on boundary_marker_
  mimi::utils::Vector<int> marked_boundary_elements_;

  /// basic ctor saved ptr to precomputed and name to use as key in precomputed
  NonlinearBase(
      const std::string& name,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : name_(name),
        precomputed_(precomputed) {}

  ///
  virtual const std::string& Name() const { return name_; }

  /// NThread domain assemble
  virtual void AssembleDomainResidual(const mfem::Vector& current_x) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("AssembleDomainResidual not implemented");
  }

  /// NThread domain grad assemble
  virtual void AssembleDomainGrad(const mfem::Vector& current_x) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("AssembleDomainGrad not implemented");
  }

  virtual void SetBoundaryMarker(const mfem::Array<int>* b_marker) {
    MIMI_FUNC()

    boundary_marker_ = b_marker;
  }

  /// NThread boundary assemble
  virtual void AssembleBoundaryResidual(const mfem::Vector& current_x) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("AssembleBoundaryResidual not implemented");
  }

  /// NThread boundary grad assemble
  virtual void AssembleBoundaryGrad(const mfem::Vector& current_x) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("AssembleBoundaryGrad not implemented");
  }

  virtual void UpdateLagrange() {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("UpdateLagrange not implemented for",
                                    Name());
  }

  virtual void FillLagrange(const double value) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("FillLagrange not implemented for", Name());
  }

  /// implemement criterium in case norm(residual) == 0 is not enough
  /// default is true;
  virtual bool Satisfied() const {
    MIMI_FUNC()

    return true;
  }
};

} // namespace mimi::integrators
