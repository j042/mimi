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
  constexpr static const int kMaxTrueDof = 100;
  constexpr static const int kDimDim = 9;

public:
  using Base_ = NonlinearSolid;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;

  struct TemporaryData : Base_::TemporaryData {
    using BaseTD_ = Base_::TemporaryData;
    using BaseTD_::dN_dx_;
    using BaseTD_::element_x_;     // x
    using BaseTD_::element_x_mat_; // x as matrix
    using BaseTD_::F_;
    using BaseTD_::F_inv_;
    using BaseTD_::forward_residual_;
    using BaseTD_::stress_;

    mfem::Vector element_v_;          // v
    mfem::DenseMatrix element_v_mat_; // v as matrix
    mfem::DenseMatrix F_dot_;

    void SetShape(const int n_dof, const int dim) {
      MIMI_FUNC()

      BaseTD_::SetShape(n_dof, dim);

      element_v_.SetSize(n_dof * dim);
      element_v_mat_.UseExternalData(element_v_.GetData(), n_dof, dim);
      F_dot_.SetSize(n_dof, dim);
    }

    void CurrentElementSolutionCopy(const mfem::Vector& all_x,
                                    const mfem::Vector& all_v,
                                    const ElementData& elem_data) {
      MIMI_FUNC()

      const double* all_x_data = all_x.GetData();
      const double* all_v_data = all_v.GetData();

      double* elem_x_data = element_x_.GetData();
      double* elem_v_data = element_v_.GetData();

      for (const int& vdof : *elem_data.v_dofs_) {
        *elem_x_data++ = all_x_data[vdof];
        *elem_v_data++ = all_v_data[vdof];
      }
    }
  };

  /// inherit ctor
  using Base_::Base_;

  /// This one needs / stores
  /// - basis(shape) function derivative (at reference)
  /// - reference to target jacobian weight (det)
  /// - target to reference jacobian (inverse of previous)
  /// - basis derivative at target
  virtual void Prepare(const int quadrature_order = -1) {
    MIMI_FUNC()

    Base_::Prepare(quadrature_order);
  }

  /// Performs quad loop with element data and temporary data
  void QuadLoop(const mfem::DenseMatrix& x,
                const mfem::DenseMatrix& v,
                const int i_thread,
                const Vector_<QuadData>& q_data,
                TemporaryData& tmp,
                mfem::DenseMatrix& residual_matrix) const {
    MIMI_FUNC()
    const bool frozen = *Base_::operator_frozen_state_;

    for (const QuadData& q : q_data) {
      // get dx_dX = x * dN_dX
      mfem::MultAtB(x, q.dN_dX_, tmp.F_);
      mfem::MultAtB(v, q.dN_dX_, tmp.F_dot_);

      // currently we will just use PK1
      material_->EvaluatePK1(tmp.F_,
                             tmp.F_dot_,
                             i_thread,
                             q.material_state_,
                             tmp.stress_,
                             frozen);

      mfem::AddMult_a_ABt(q.integration_weight_ * q.det_dX_dxi_,
                          q.dN_dX_,
                          tmp.stress_,
                          residual_matrix);
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
      TemporaryData tmp;
      for (int i{begin}; i < end; ++i) {
        // in
        const ElementData& e = element_data_[i];
        // set shape for tmp data - first call will allocate
        tmp.SetShape(e.n_dof_, dim_);
        // variable name is misleading - this is just local residual
        // we use this container, as we already allocate this in tmp anyways
        tmp.forward_residual_ = 0.0;

        // get current element solution as matrix
        tmp.CurrentElementSolutionCopy(current_x, current_v, e);

        // get element state view
        mfem::DenseMatrix& current_element_x = tmp.element_x_mat_;
        mfem::DenseMatrix& current_element_v = tmp.element_v_mat_;

        // assemble residual
        QuadLoop(current_element_x,
                 current_element_v,
                 i_thread,
                 e.quad_data_,
                 tmp,
                 tmp.forward_residual_);

        // push right away - seems to work quite well!
        const std::lock_guard<std::mutex> lock(residual_mutex);
        residual.AddElementVector(*e.v_dofs_, tmp.forward_residual_.GetData());
      }
    };
    mimi::utils::NThreadExe(assemble_element_residual_and_contribute,
                            n_elements_,
                            (nthreads < 1) ? n_threads_ : nthreads);
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
          TemporaryData tmp;
          mfem::DenseMatrix local_residual;
          mfem::DenseMatrix local_grad;
          for (int i{begin}; i < end; ++i) {
            // in
            const ElementData& e = element_data_[i];
            local_residual.SetSize(e.n_dof_, dim_);
            local_residual = 0.0;
            local_grad.SetSize(e.n_tdof_, e.n_tdof_);

            // set shape for tmp data - first call will allocate
            tmp.SetShape(e.n_dof_, dim_);

            // get element state view
            tmp.CurrentElementSolutionCopy(current_x, current_v, e);
            mfem::DenseMatrix& current_element_x = tmp.element_x_mat_;
            mfem::DenseMatrix& current_element_v = tmp.element_v_mat_;

            // assemble residual
            QuadLoop(current_element_x,
                     current_element_v,
                     i_thread,
                     e.quad_data_,
                     tmp,
                     local_residual);

            double* grad_data = local_grad.GetData();
            double* solution_data = current_element_x.GetData();
            const double* residual_data = local_residual.GetData();
            const double* fd_forward_data = tmp.forward_residual_.GetData();
            for (int j{}; j < e.n_tdof_; ++j) {
              tmp.forward_residual_ = 0.0;

              double& with_respect_to = *solution_data++;
              const double orig_wrt = with_respect_to;
              const double diff_step = std::abs(orig_wrt) * 1.0e-8;
              const double diff_step_inv = 1. / diff_step;

              with_respect_to = orig_wrt + diff_step;
              QuadLoop(current_element_x,
                       current_element_v,
                       i_thread,
                       e.quad_data_,
                       tmp,
                       tmp.forward_residual_);

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
            const double* local_A = local_grad.GetData();
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
          TemporaryData tmp;
          mfem::DenseMatrix local_residual;
          mfem::DenseMatrix local_grad;
          for (int i{begin}; i < end; ++i) {
            // in
            const ElementData& e = element_data_[i];

            local_residual.SetSize(e.n_dof_, dim_);
            local_grad.SetSize(e.n_tdof_, e.n_tdof_);
            local_residual = 0.0;

            // set shape for tmp data - first call will allocate
            tmp.SetShape(e.n_dof_, dim_);

            // get element state view
            tmp.CurrentElementSolutionCopy(current_x, current_v, e);
            mfem::DenseMatrix& current_element_x = tmp.element_x_mat_;
            mfem::DenseMatrix& current_element_v = tmp.element_v_mat_;

            // assemble residual
            QuadLoop(current_element_x,
                     current_element_v,
                     i_thread,
                     e.quad_data_,
                     tmp,
                     local_residual);

            double* grad_data = local_grad.GetData();
            double* solution_data = current_element_x.GetData();
            const double* residual_data = local_residual.GetData();
            const double* fd_forward_data = tmp.forward_residual_.GetData();
            for (int j{}; j < e.n_tdof_; ++j) {
              tmp.forward_residual_ = 0.0;

              double& with_respect_to = *solution_data++;
              const double orig_wrt = with_respect_to;
              const double diff_step = std::abs(orig_wrt) * 1.0e-8;
              const double diff_step_inv = 1. / diff_step;

              with_respect_to = orig_wrt + diff_step;
              QuadLoop(current_element_x,
                       current_element_v,
                       i_thread,
                       e.quad_data_,
                       tmp,
                       tmp.forward_residual_);
              for (int k{}; k < e.n_tdof_; ++k) {
                *grad_data++ =
                    (fd_forward_data[k] - residual_data[k]) * diff_step_inv;
              }
              with_respect_to = orig_wrt;
            }

            // push right away
            std::lock_guard<std::mutex> lock(residual_mutex);
            const auto& vdofs = *e.v_dofs_;
            residual.AddElementVector(vdofs, local_residual.GetData());
            double* A = grad.GetData();
            const double* local_A = local_grad.GetData();
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
                               mfem::Vector& residual) {
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
