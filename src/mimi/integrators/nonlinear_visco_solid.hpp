#pragma once

#include "mimi/integrators/materials.hpp"
#include "mimi/integrators/nonlinear_solid.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/n_thread_exe.hpp"

namespace mimi::integrators {

/// basic integrator of nonlinear solids
/// given current x coordinate (NOT displacement)
/// Computes F and passes it to material
class NonlinearViscoSolid : public NonlinearSolid {
public:
  using Base_ = NonlinearSolid;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;

private:
  mutable Vector_<TemporaryViscoData> temporary_data_;

public:
  /// inherit ctor
  using Base_::Base_;

  virtual void PrepareTemporaryDataAndMaterial() {
    MIMI_FUNC()

    temporary_data_.resize(n_threads_);
    for (auto& td : temporary_data_) {
      td.SetDim(dim_);
      material_->AllocateAux(td);
    }
  }

  /// This one needs / stores
  /// - basis(shape) function derivative (at reference)
  /// - reference to target jacobian weight (det)
  /// - target to reference jacobian (inverse of previous)
  /// - basis derivative at target
  virtual void Prepare() {
    MIMI_FUNC()

    Base_::Prepare();
  }

  /// Performs quad loop with element data and temporary data
  void QuadLoop(const Vector_<QuadData>& q_data,
                TemporaryViscoData& tmp,
                mfem::DenseMatrix& residual_matrix) const {
    MIMI_FUNC()
    for (const QuadData& q : q_data) {
      tmp.ComputeFAndFDot(q.dN_dX_);

      // currently we will just use PK1
      material_->EvaluatePK1(q.material_state_,
                             tmp,
                             tmp.stress_); // quad loop is always frozen

      mfem::AddMult_a_ABt(q.integration_weight_ * q.det_dX_dxi_,
                          q.dN_dX_,
                          tmp.stress_,
                          residual_matrix);
    }
  }

  void AccumulateStatesAtQuads(Vector_<QuadData>& q_data,
                               TemporaryViscoData& tmp) const {
    MIMI_FUNC()

    for (QuadData& q : q_data) {
      tmp.ComputeFAndFDot(q.dN_dX_);

      // currently we will just use PK1
      material_->Accumulate(q.material_state_, tmp);
    }
  }

  virtual void AddDomainResidual(const mfem::Vector& current_x,
                                 const mfem::Vector& current_v,
                                 const int nthreads,
                                 mfem::Vector& residual) const {
    material_->dt_ = dt_;
    material_->first_effective_dt_ = first_effective_dt_;
    material_->second_effective_dt_ = second_effective_dt_;

    std::mutex residual_mutex;
    // lambda for nthread assemble
    auto assemble_element_residual_and_contribute = [&](const int begin,
                                                        const int end,
                                                        const int i_thread) {
      auto& tmp = temporary_data_[i_thread];
      for (int i{begin}; i < end; ++i) {
        // in
        const ElementData& e = element_data_[i];
        // set shape for tmp data - first call will allocate
        tmp.SetDof(e.n_dof_);
        // variable name is misleading - this is just local residual
        // we use this container, as we already allocate this in tmp anyways
        tmp.local_residual_ = 0.0;

        // get current element solution as matrix
        tmp.CurrentElementSolutionCopy(current_x, current_v, *e.v_dofs_);

        // assemble residual
        QuadLoop(e.quad_data_, tmp, tmp.local_residual_);

        // push right away - seems to work quite well!
        const std::lock_guard<std::mutex> lock(residual_mutex);
        residual.AddElementVector(*e.v_dofs_, tmp.local_residual_.GetData());
      }
    };
    mimi::utils::NThreadExe(assemble_element_residual_and_contribute,
                            n_elements_,
                            (nthreads < 1) ? n_threads_ : nthreads);
  }

  virtual void DomainPostTimeAdvance(const mfem::Vector& current_x) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "Visco solids needs current_v for post time step processing");
  }

  virtual void DomainPostTimeAdvance(const mfem::Vector& current_x,
                                     const mfem::Vector& current_v) {
    MIMI_FUNC()
    if (!has_states_)
      return;
    auto accumulate_states =
        [&](const int begin, const int end, const int i_thread) {
          auto& tmp = temporary_data_[i_thread];

          for (int i{begin}; i < end; ++i) {
            // in
            ElementData& e = element_data_[i];
            // set shape for tmp data - first call will allocate
            tmp.SetDof(e.n_dof_);

            // get current element solution as matrix
            tmp.CurrentElementSolutionCopy(current_x, current_v, *e.v_dofs_);

            // accumulate
            AccumulateStatesAtQuads(e.quad_data_, tmp);
          }
        };
    mimi::utils::NThreadExe(accumulate_states, n_elements_, n_threads_);

    auto& rc = *RuntimeCommunication();
    if (rc.ShouldSave("temperature")) {
      projected_.SetSize(precomputed_->n_v_dofs_);
      Temperature(projected_);
      rc.SaveDynamicVector("temperature_", projected_);
    }
  }

  virtual void BoundaryPostTimeAdvance(const mfem::Vector& current_x) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "Visco solids needs current_v for post time step processing");
  }

  virtual void BoundaryPostTimeAdvance(const mfem::Vector& current_x,
                                       const mfem::Vector& current_v) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "NonlinearViscoSolid::BoundaryPostTimeAdvance not implemented");
  }

  virtual void AddDomainGrad(const mfem::Vector& current_x,
                             const mfem::Vector& current_v,
                             const int nthreads,
                             mfem::SparseMatrix& grad) const {
    material_->dt_ = dt_;
    material_->first_effective_dt_ = first_effective_dt_;
    material_->second_effective_dt_ = second_effective_dt_;

    std::mutex residual_mutex;
    // lambda for nthread assemble
    auto assemble_element_residual_and_grad_then_contribute =
        [&](const int begin, const int end, const int i_thread) {
          auto& tmp = temporary_data_[i_thread];
          for (int i{begin}; i < end; ++i) {
            // in
            const ElementData& e = element_data_[i];
            // set shape for tmp data - first call will allocate
            tmp.SetDof(e.n_dof_);
            tmp.local_residual_ = 0.0;

            // get element state view
            tmp.CurrentElementSolutionCopy(current_x, current_v, *e.v_dofs_);
            mfem::DenseMatrix& current_element_x = tmp.element_x_mat_;

            // assemble residual
            QuadLoop(e.quad_data_, tmp, tmp.local_residual_);

            double* grad_data = tmp.local_grad_.GetData();
            double* solution_data = current_element_x.GetData();
            const double* residual_data = tmp.local_residual_.GetData();
            const double* fd_forward_data = tmp.forward_residual_.GetData();
            for (int j{}; j < e.n_tdof_; ++j) {
              tmp.forward_residual_ = 0.0;

              double& with_respect_to = *solution_data++;
              const double orig_wrt = with_respect_to;
              const double diff_step = std::abs(orig_wrt) * 1.0e-8;
              const double diff_step_inv = 1. / diff_step;

              with_respect_to = orig_wrt + diff_step;
              QuadLoop(e.quad_data_, tmp, tmp.forward_residual_);

              for (int k{}; k < e.n_tdof_; ++k) {
                *grad_data++ =
                    (fd_forward_data[k] - residual_data[k]) * diff_step_inv;
              }
              with_respect_to = orig_wrt;
            }

            // push right away
            std::lock_guard<std::mutex> lock(residual_mutex);
            const auto& vdofs = *e.v_dofs_;
            double* A = grad.GetData();
            const double* local_A = tmp.local_grad_.GetData();
            const auto& A_ids = *precomputed_->domain_A_ids_[i];
            for (int k{}; k < A_ids.size(); ++k) {
              A[A_ids[k]] += *local_A++;
            }
          }
        };

    mimi::utils::NThreadExe(assemble_element_residual_and_grad_then_contribute,
                            n_elements_,
                            (nthreads < 0) ? n_threads_ : nthreads);
  };

  virtual void AddDomainResidualAndGrad(const mfem::Vector& current_x,
                                        const mfem::Vector& current_v,
                                        const int nthreads,
                                        const double grad_factor,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) const {
    material_->dt_ = dt_;
    material_->first_effective_dt_ = first_effective_dt_;
    material_->second_effective_dt_ = second_effective_dt_;

    std::mutex residual_mutex;
    // lambda for nthread assemble
    auto assemble_element_residual_and_grad_then_contribute =
        [&](const int begin, const int end, const int i_thread) {
          auto& tmp = temporary_data_[i_thread];
          for (int i{begin}; i < end; ++i) {
            // in
            const ElementData& e = element_data_[i];
            // set shape for tmp data - first call will allocate
            tmp.SetDof(e.n_dof_);
            tmp.local_residual_ = 0.0;

            // get element state view
            tmp.CurrentElementSolutionCopy(current_x, current_v, *e.v_dofs_);
            mfem::DenseMatrix& current_element_x = tmp.element_x_mat_;

            // assemble residual
            QuadLoop(e.quad_data_, tmp, tmp.local_residual_);

            double* grad_data = tmp.local_grad_.GetData();
            double* solution_data = current_element_x.GetData();
            const double* residual_data = tmp.local_residual_.GetData();
            const double* fd_forward_data = tmp.forward_residual_.GetData();
            for (int j{}; j < e.n_tdof_; ++j) {
              tmp.forward_residual_ = 0.0;

              double& with_respect_to = *solution_data++;
              const double orig_wrt = with_respect_to;
              const double diff_step = std::abs(orig_wrt) * 1.0e-8;
              const double diff_step_inv = 1. / diff_step;

              with_respect_to = orig_wrt + diff_step;
              QuadLoop(e.quad_data_, tmp, tmp.forward_residual_);
              for (int k{}; k < e.n_tdof_; ++k) {
                *grad_data++ =
                    (fd_forward_data[k] - residual_data[k]) * diff_step_inv;
              }
              with_respect_to = orig_wrt;
            }

            // push right away
            std::lock_guard<std::mutex> lock(residual_mutex);
            const auto& vdofs = *e.v_dofs_;
            residual.AddElementVector(vdofs, tmp.local_residual_.GetData());
            double* A = grad.GetData();
            const double* local_A = tmp.local_grad_.GetData();
            const auto& A_ids = *precomputed_->domain_A_ids_[i];
            for (int k{}; k < A_ids.size(); ++k) {
              A[A_ids[k]] += *local_A++ * grad_factor;
            }
          }
        };

    mimi::utils::NThreadExe(assemble_element_residual_and_grad_then_contribute,
                            n_elements_,
                            (nthreads < 0) ? n_threads_ : nthreads);
  }

  virtual void AddBoundaryResidual(const mfem::Vector& current_x,
                                   const mfem::Vector& current_v,
                                   const int nthreads,
                                   mfem::Vector& residual) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("mimi::integrators::NonlinearViscoSolid::"
                                    "AddBoundaryResidual not implemented");
  }

  virtual void AddBoundaryGrad(const mfem::Vector& current_x,
                               const mfem::Vector& current_v,
                               const int nthreads,
                               mfem::SparseMatrix& residual) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("mimi::integrators::NonlinearViscoSolid::"
                                    "AddBoundaryGrad not implemented");
  }

  virtual void AddBoundaryResidualAndGrad(const mfem::Vector& current_x,
                                          const mfem::Vector& current_v,
                                          const int nthreads,
                                          const double grad_factor,
                                          mfem::Vector& residual,
                                          mfem::SparseMatrix& grad) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "mimi::integrators::NonlinearViscoSolid::"
        "AddBoundaryResidualAndGrad not implemented");
  }
};

} // namespace mimi::integrators
