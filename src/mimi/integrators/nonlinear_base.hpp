#pragma once

#include <cmath>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/utils/containers.hpp"
#include "mimi/utils/precomputed.hpp"
#include "mimi/utils/print.hpp"
#include "mimi/utils/runtime_communication.hpp"

namespace mimi::integrators {

class NonlinearBase : public mfem::NonlinearFormIntegrator {
public:
  std::shared_ptr<mimi::utils::PrecomputedData> precomputed_;
  std::unique_ptr<mimi::utils::Data<mfem::Vector>> element_vectors_;
  std::unique_ptr<mimi::utils::Data<mfem::Vector>> boundary_element_vectors_;
  std::unique_ptr<mimi::utils::Data<mfem::DenseMatrix>> element_matrices_;
  std::unique_ptr<mimi::utils::Data<mfem::DenseMatrix>>
      boundary_element_matrices_;

  std::shared_ptr<mimi::utils::RuntimeCommunication> runtime_communication_;

  std::string name_;

  /// flag to know when to assemble grad.
  bool assemble_grad_{false};

  /// time step size, in case you need them
  /// nonlinear forms will set them
  double dt_{0.0};
  double first_effective_dt_{0.0};  // this is for x
  double second_effective_dt_{0.0}; // this is for x_dot (=v).

  /// marker for this bdr face integ
  const mfem::Array<int>* boundary_marker_ = nullptr;

  /// ids of contributing boundary elements, extracted based on boundary_marker_
  mimi::utils::Vector<int> marked_boundary_elements_;

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

  virtual std::shared_ptr<mimi::utils::RuntimeCommunication>
  RuntimeCommunication() {
    MIMI_FUNC()
    if (!runtime_communication_) {
      mimi::utils::PrintAndThrowError("runtime_communication_ not set.");
    }
    return runtime_communication_;
  }

  /// Precompute call interface
  virtual void Prepare() {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("Prepare not implemented");
  };

  virtual void AddDomainResidual(const mfem::Vector& current_x,
                                 const int nthreads,
                                 mfem::Vector& residual) const {
    mimi::utils::PrintAndThrowError("AddDomainResidual not implemented");
  };

  virtual void AddDomainGrad(const mfem::Vector& current_x,
                             const int nthreads,
                             mfem::SparseMatrix& grad) const {
    mimi::utils::PrintAndThrowError("AddDomainGrad not implemented");
  };

  virtual void AddDomainResidualAndGrad(const mfem::Vector& current_x,
                                        const int nthreads,
                                        const double grad_factor,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) const {
    mimi::utils::PrintAndThrowError("AddDomainResidualAndGrad not implemented");
  };

  virtual void DomainPostTimeAdvance(const mfem::Vector& current_x) {
    mimi::utils::PrintAndThrowError("DomainPostTimeAdvance not implemented");
  }

  virtual void AddBoundaryResidual(const mfem::Vector& current_x,
                                   const int nthreads,
                                   mfem::Vector& residual) {
    mimi::utils::PrintAndThrowError("AddBoundaryResidual not implemented");
  };

  virtual void AddBoundaryGrad(const mfem::Vector& current_x,
                               const int nthreads,
                               mfem::SparseMatrix& grad) {
    mimi::utils::PrintAndThrowError("AddBoundaryGrad not implemented");
  };

  virtual void AddBoundaryResidualAndGrad(const mfem::Vector& current_x,
                                          const int nthreads,
                                          const double grad_factor,
                                          mfem::Vector& residual,
                                          mfem::SparseMatrix& grad) {
    mimi::utils::PrintAndThrowError(
        "AddBoundaryResidualAndGrad not implemented");
  };

  virtual void BoundaryPostTimeAdvance(const mfem::Vector& current_x) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("BoundaryPostTimeAdvance not implemented");
  }

  virtual void SetBoundaryMarker(const mfem::Array<int>* b_marker) {
    MIMI_FUNC()

    boundary_marker_ = b_marker;
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

  virtual double GapNorm(const mfem::Vector& test_x, const int nthreads) const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("GapNorm(x, nthread) not implemented for",
                                    Name());
    return 0.0;
  }

  virtual double LastGapNorm() const {
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

  virtual void Temperature(mfem::Vector& integrated) {
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

/// Currently, we don't split hpp and cpp, so we move nested structure to a
/// mutual place, here.
/// temporary containers required in element assembly. one for each thread
struct TemporaryData {
  int dim_;

  // state variable
  double det_F_{};
  bool has_det_F_{false};
  bool has_F_inv_{false};

  // general assembly items
  mfem::Vector element_x_;
  mfem::DenseMatrix element_x_mat_;
  mfem::DenseMatrix local_residual_;
  mfem::DenseMatrix local_grad_;
  mfem::DenseMatrix forward_residual_;

  // used for materials
  mfem::DenseMatrix stress_;
  mfem::DenseMatrix F_;
  mfem::DenseMatrix F_inv_;
  mfem::DenseMatrix F_dot_; // for visco but put it here for easier visibility

  // used in materials - materials will visit and initiate those in
  // PrepareTemporaryData
  mfem::DenseMatrix I_;
  mfem::DenseMatrix alternative_stress_;           // for conversion
  mimi::utils::Vector<mfem::Vector> aux_vec_;      // for computation
  mimi::utils::Vector<mfem::DenseMatrix> aux_mat_; // for computation

  /// this can be called once at Prepare()
  void SetDim(const int dim) {
    MIMI_FUNC()
    dim_ = dim;

    stress_.SetSize(dim, dim);
    F_.SetSize(dim, dim);
    F_inv_.SetSize(dim, dim);
    F_dot_.SetSize(dim, dim);

    I_.Diag(1., dim);
    alternative_stress_.SetSize(dim, dim);
  }

  /// this should be called at the start of every element as NDof may change
  void Reset(const int n_dof) {
    MIMI_FUNC()
    has_det_F_ = false;
    has_F_inv_ = false;

    element_x_.SetSize(n_dof * dim_); // will be resized in getsubvector
    element_x_mat_.UseExternalData(element_x_.GetData(), n_dof, dim_);
    forward_residual_.SetSize(n_dof, dim_);
    local_residual_.SetSize(n_dof, dim_);
    local_grad_.SetSize(n_dof * dim_, n_dof * dim_);
  }

  mfem::DenseMatrix& FInv() {
    MIMI_FUNC()
    if (has_F_inv_) {
      return F_inv_;
    }
    mfem::CalcInverse(F_, F_inv_);

    has_F_inv_ = true;
    return F_inv_;
  }

  double DetF() {
    MIMI_FUNC()
    if (has_det_F_) {
      return det_F_;
    }

    det_F_ = F_.Det();
    has_det_F_ = true;
    return det_F_;
  }

  mfem::DenseMatrix& CurrentElementSolutionCopy(const mfem::Vector& current_all,
                                                const mfem::Array<int>& vdofs) {
    MIMI_FUNC()

    current_all.GetSubVector(vdofs, element_x_);
    return element_x_mat_;
  }
};

struct TemporaryViscoData : TemporaryData {
  using BaseTD_ = TemporaryData;

  mfem::Vector element_v_;          // v
  mfem::DenseMatrix element_v_mat_; // v as matrix

  using BaseTD_::FInv;
  using BaseTD_::DetF;

  void Reset(const int n_dof) {
    MIMI_FUNC()

    BaseTD_::Reset(n_dof);

    element_v_.SetSize(n_dof * dim_);
    element_v_mat_.UseExternalData(element_v_.GetData(), n_dof, dim_);
  }

  void CurrentElementSolutionCopy(const mfem::Vector& all_x,
                                  const mfem::Vector& all_v,
                                  const mfem::Array<int>& vdofs) {
    MIMI_FUNC()

    const double* all_x_data = all_x.GetData();
    const double* all_v_data = all_v.GetData();

    double* elem_x_data = element_x_.GetData();
    double* elem_v_data = element_v_.GetData();

    for (const int& vdof : vdofs) {
      *elem_x_data++ = all_x_data[vdof];
      *elem_v_data++ = all_v_data[vdof];
    }
  }
};

} // namespace mimi::integrators
