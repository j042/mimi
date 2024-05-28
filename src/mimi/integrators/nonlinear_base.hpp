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

  /// flag to know when to assembly grad.
  bool assemble_grad_{false};
  /// flag to know when not to accumulate
  bool frozen_state_{false};

  /// time step size, in case you need them
  /// nonlinear forms will set them
  double dt_{0.0};
  double first_effective_dt_{0.0};  // this is for x
  double second_effective_dt_{0.0}; // this is for x_dot (=v).

  /// marker for this bdr face integ
  const mfem::Array<int>* boundary_marker_ = nullptr;

  /// ids of contributing boundary elements, extracted based on boundary_marker_
  mimi::utils::Vector<int> marked_boundary_elements_;

  /// @brief quadrature order per elements - alternatively, could be per
  /// patch thing
  mimi::utils::Vector<int> quadrature_orders_;

  /// @brief quadrature order per boundary elements - alternatively, could be
  /// per boundary patch thing
  mimi::utils::Vector<int> boundary_quadrature_orders_;

  /// decided to ask for intrules all the time. but since we don't wanna call
  /// the geometry type all the time, we save this just once.
  mfem::Geometry::Type geometry_type_;

  /// see geometry_type_
  mfem::Geometry::Type boundary_geometry_type_;

  /// convenient constants - space dim
  int dim_;

  /// basic ctor saved ptr to precomputed and name to use as key in precomputed
  NonlinearBase(
      const std::string& name,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : name_(name),
        precomputed_(precomputed) {}

  /// Name of the integrator
  virtual const std::string& Name() const { return name_; }

  /// Precompute call interface
  virtual void Prepare(const int quadrature_order) {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("Prepare not implemented");
  };

  virtual void AddDomainResidual(const mfem::Vector& current_x,
                                 mfem::Vector& residual) const;
  virtual void AddDomainResidualAndGrad(const mfem::Vector& current_x,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) const;

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

  /// boundaries are kept
  virtual void
  AddToGlobalBoundaryResidual(mfem::Vector& global_residual) const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "AddToGlobalBoundaryResidual not implemented for",
        Name());
  }

  virtual void AddToGlobalBoundaryGrad(mfem::SparseMatrix& global_grad) const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "AddToGlobalBoundaryGrad not implemented for",
        Name());
  }
  virtual double GapNorm() const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("GapNorm not implemented for", Name());
    return 0.0;
  }

  virtual void AccumulatedPlasticStrain(mfem::Vector& x,
                                        mfem::Vector& integrated) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "AccumulatedPlasticStrain not implemented for",
        Name());
  }

  virtual void Temperature(mfem::Vector& x, mfem::Vector& integrated) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("Temperature not implemented for", Name());
  }

  virtual void VonMisesStress(mfem::Vector& x, mfem::Vector& integrated) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("VonMisesStress not implemented for",
                                    Name());
  }

  virtual void SigmaXX(mfem::Vector& x, mfem::Vector& integrated) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("SigmaXX not implemented for", Name());
  }

  virtual void SigmaYY(mfem::Vector& x, mfem::Vector& integrated) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("SigmaYY not implemented for", Name());
  }

  virtual void SigmaZZ(mfem::Vector& x, mfem::Vector& integrated) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("SigmaZZ not implemented for", Name());
  }
};

} // namespace mimi::integrators
