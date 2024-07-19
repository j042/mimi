#pragma once

#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/materials/materials.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/n_thread_exe.hpp"
#include "mimi/utils/precomputed.hpp"

namespace mimi::integrators {

/// Temporary containers required in element assembly. One for each thread will
/// be created. Stores values required for material, quad point and element
/// data.
class NonlinearSolidWorkData {
public:
  // basic info. used to compute tdofs
  int dim_;
  int n_dof_;

  // state variable
  double det_F_{};
  bool has_det_F_{false};
  bool has_F_inv_{false};

  /// flag to inform residual destination
  bool assembling_grad_{false};

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
  // PrepareWorkData
  mfem::DenseMatrix I_;
  mfem::DenseMatrix alternative_stress_;           // for conversion
  mimi::utils::Vector<mfem::Vector> aux_vec_;      // for computation
  mimi::utils::Vector<mfem::DenseMatrix> aux_mat_; // for computation

  // this is global sized local residual / grad entries
  mfem::Vector thread_local_residual_;
  mfem::Vector thread_local_A_;

  /// this can be called once at Prepare()
  void SetDim(const int dim);

  /// this should be called at the start of every element as NDof may change
  void SetDof(const int n_dof);

  /// Number of true dof based on dim and n_dof values
  int GetTDof() const { return dim_ * n_dof_; }

  /// computes F, deformation gradient, and resets flags
  void ComputeF(const mfem::DenseMatrix& dNdX);

  /// Returns inverse of F. For consecutive calls, it will return stored value
  mfem::DenseMatrix& FInv() {
    MIMI_FUNC()
    if (has_F_inv_) {
      return F_inv_;
    }
    mfem::CalcInverse(F_, F_inv_);

    has_F_inv_ = true;
    return F_inv_;
  }

  /// determinant of F
  double DetF() {
    MIMI_FUNC()
    if (has_det_F_) {
      return det_F_;
    }

    det_F_ = F_.Det();
    has_det_F_ = true;
    return det_F_;
  }

  /// Given global state vector and support ids, copies element vector and
  /// returns matrix form.
  mfem::DenseMatrix& CurrentElementSolutionCopy(const mfem::Vector& current_all,
                                                const mfem::Array<int>& vdofs);

  mfem::DenseMatrix& CurrentSolution() {
    MIMI_FUNC()

    return element_x_mat_;
  }

  /// hint flag to inform current action. influences return value for
  /// ResidualMatrix()
  void GradAssembly(bool state) {
    MIMI_FUNC()

    assembling_grad_ = state;
  }

  /// Returns destination matrix for assembly. This maybe for residual or FD.
  mfem::DenseMatrix& ResidualMatrix() {
    MIMI_FUNC()
    if (assembling_grad_) {
      return forward_residual_;
    }
    return local_residual_;
  }
};

/// basic integrator of nonlinear solids
/// given current x coordinate (NOT displacement)
/// Computes F and passes it to material
class NonlinearSolid : public NonlinearBase {
public:
  /// used to project
  std::unique_ptr<mfem::SparseMatrix> m_mat_;
  mfem::CGSolver mass_inv_;
  mfem::DSmoother mass_inv_prec_;
  mfem::UMFPackSolver mass_inv_direct_;
  mfem::Vector integrated_;
  mfem::Vector projected_; /* optional use */

  using Base_ = NonlinearBase;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;
  using ElementQuadData_ = mimi::utils::ElementQuadData;
  using ElementData_ = mimi::utils::ElementData;
  using QuadData_ = mimi::utils::QuadData;

protected:
  /// number of threads for this system
  int n_threads_;
  int n_elements_;
  /// material related
  std::shared_ptr<mimi::materials::MaterialBase> material_;
  // no plan for mixed material, so we can have this flag in integrator level
  bool has_states_;

private:
  mutable Vector_<NonlinearSolidWorkData> work_data_;

public:
  NonlinearSolid(
      const std::string& name,
      const std::shared_ptr<mimi::materials::MaterialBase>& material,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : NonlinearBase(name, precomputed),
        material_{material} {}

  virtual const std::string& Name() const { return name_; }

  virtual void PrepareNonlinearSolidWorkDataAndMaterial();

  /// This one needs / stores
  /// - basis(shape) function derivative (at reference)
  /// - reference to target jacobian weight (det)
  /// - target to reference jacobian (inverse of previous)
  /// - basis derivative at target
  virtual void Prepare();

  /// reusable element assembly routine. Make sure CurrentElementSolutionCopy is
  /// called beforehand. Specify residual destination to enable fd
  template<bool accumulate_state_only = false>
  void ElementResidual(std::conditional_t<accumulate_state_only,
                                          Vector_<QuadData_>,
                                          const Vector_<QuadData_>>& q_data,
                       NonlinearSolidWorkData& w) const {
    MIMI_FUNC()

    w.ResidualMatrix() = 0.0;

    // quad loop - we assume w.element_x_mat_ is prepared
    for (auto& q : q_data) {
      w.ComputeF(q.dN_dX);
      if constexpr (!accumulate_state_only) {
        material_->EvaluatePK1(q.material_state, w, w.stress_);
        mfem::AddMult_a_ABt(q.integration_weight * q.det_dX_dxi,
                            q.dN_dX,
                            w.stress_,
                            w.ResidualMatrix());
      } else {
        material_->Accumulate(q.material_state, w);
      }
    }
  }

  /// reusable element residual and jacobian assembly. Make sure
  /// CurrentElementSolutionCopy is called beforehand
  void ElementResidualAndGrad(const Vector_<QuadData_>& q_data,
                              NonlinearSolidWorkData& w) const;

  /// Each thread assembles residual in a separate vector to avoid race
  /// condition. This can be then reduced to one vector using
  /// AddThreadLocalResidual()
  void ThreadLocalResidual(const mfem::Vector& current_u) const;

  /// Similar to ThreadLocalResidual, but also grad
  void ThreadLocalResidualAndGrad(const mfem::Vector& current_u,
                                  const int A_nnz) const;

  /// Adds residual contribution
  virtual void AddDomainResidual(const mfem::Vector& current_u,
                                 mfem::Vector& residual) const;

  /// Adds jacobian contribution. Currently not used, as it is more efficient to
  /// assemble residual and grad at the same time using computational
  /// differentiation.
  virtual void AddDomainGrad(const mfem::Vector& current_u,
                             mfem::SparseMatrix& grad) const {

    mimi::utils::PrintAndThrowError(
        "Currently not implemented, use AddDomainResidualAndGrad");
  }

  /// Adds residual jacobian contribution
  virtual void AddDomainResidualAndGrad(const mfem::Vector& current_u,
                                        const double grad_factor,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) const;

  /// using converged solution at the time step, postprocess. For example,
  /// actual accumulation of material states.
  virtual void DomainPostTimeAdvance(const mfem::Vector& converged_x);
};

} // namespace mimi::integrators
