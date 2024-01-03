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
  constexpr static const int kMaxTrueDof = 50;
  constexpr static const int kDimDim = 9;

public:
  using Base_ = NonlinearSolid;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;

  struct TemporaryData : Base_::TemporaryData {
    using BaseTD_ = Base_::TemporaryData;
    using BaseTD_::backward_residual_;
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

    /// @brief
    /// @param element_x_data
    /// @param stress_data
    /// @param dN_dx_data
    /// @param F_data
    /// @param F_inv_data
    /// @param F_dot_data
    /// @param dim
    void SetData(double* element_x_data,
                 double* element_v_data,
                 double* stress_data,
                 double* dN_dx_data,
                 double* F_data,
                 double* F_inv_data,
                 double* F_dot_data,
                 double* forward_residual_data,
                 double* backward_residual_data,
                 const int dim) {
      MIMI_FUNC()

      // use base' for overlapping variables
      BaseTD_::SetData(element_x_data,
                       stress_data,
                       dN_dx_data,
                       F_data,
                       F_inv_data,
                       forward_residual_data,
                       backward_residual_data,
                       dim);

      element_v_.SetDataAndSize(element_v_data, kMaxTrueDof);
      element_v_mat_.UseExternalData(element_v_data, kMaxTrueDof, 1);
      F_dot_.UseExternalData(F_dot_data, dim, dim);
    }

    void SetShape(const int n_dof, const int dim) {
      MIMI_FUNC()

      element_x_mat_.SetSize(n_dof, dim);
      element_v_mat_.SetSize(n_dof, dim);
      dN_dx_.SetSize(n_dof, dim);
      forward_residual_.SetSize(n_dof, dim);
      backward_residual_.SetSize(n_dof, dim);
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
                Vector_<QuadData>& q_data,
                TemporaryData& tmp,
                mfem::DenseMatrix& residual_matrix) {
    MIMI_FUNC()

    for (QuadData& q : q_data) {
      // get dx_dX = x * dN_dX
      mfem::MultAtB(x, q.dN_dX_, tmp.F_);
      mfem::MultAtB(v, q.dN_dX_, tmp.F_dot_);

      // currently we will just use PK1
      material_->EvaluatePK1(tmp.F_,
                             tmp.F_dot_,
                             i_thread,
                             q.material_state_,
                             tmp.stress_);

      mfem::AddMult_a_ABt(q.integration_weight_ * q.det_dX_dxi_,
                          q.dN_dX_,
                          tmp.stress_,
                          residual_matrix);
    }
  }

  virtual void AssembleDomainResidual(const mfem::Vector& current_x,
                                      const mfem::Vector& current_v) {
    MIMI_FUNC()

    auto assemble_element_residual_and_maybe_grad =
        [&](const int begin, const int end, const int i_thread) {
          TemporaryData tmp;
          // create some space in stack
          double element_x_data[kMaxTrueDof];
          double element_v_data[kMaxTrueDof];
          double stress_data[kDimDim];
          double dN_dx_data[kMaxTrueDof];
          double F_data[kDimDim];
          double F_inv_data[kDimDim];
          double F_dot_data[kDimDim];
          double fd_forward_data[kMaxTrueDof];
          double fd_backward_data[kMaxTrueDof];
          tmp.SetData(element_x_data,
                      element_v_data,
                      stress_data,
                      dN_dx_data,
                      F_data,
                      F_inv_data,
                      F_dot_data,
                      fd_forward_data,
                      fd_backward_data,
                      dim_);

          for (int i{begin}; i < end; ++i) {
            // in
            ElementData& e = element_data_[i];
            e.residual_view_ = 0.0;

            // set shape for tmp data
            tmp.SetShape(e.n_dof_, dim_);

            // get current element solution as matrix
            tmp.CurrentElementSolutionCopy(current_x, current_v, e);

            // get element state view
            mfem::DenseMatrix& current_element_x = tmp.element_x_mat_;
            mfem::DenseMatrix& current_element_v = tmp.element_v_mat_;

            if (frozen_state_) {
              e.FreezeStates();
            } else {
              e.MeltStates();
            }

            // assembly grad with FD - currently we only change x value.
            if (assemble_grad_) {
              assert(frozen_state_);

              double* grad_data = e.grad_view_.GetData();
              double* solution_data = current_element_x.GetData();
              for (int j{}; j < e.n_tdof_; ++j) {
                tmp.forward_residual_ = 0.0;
                tmp.backward_residual_ = 0.0;

                double& with_respect_to = *solution_data++;
                const double orig_wrt = with_respect_to;
                const double diff_step = std::abs(orig_wrt) * 1.0e-8;
                const double two_diff_step_inv = 1. / (2.0 * diff_step);

                with_respect_to = orig_wrt + diff_step;
                QuadLoop(current_element_x,
                         current_element_v,
                         i_thread,
                         e.quad_data_,
                         tmp,
                         tmp.forward_residual_);

                with_respect_to = orig_wrt - diff_step;
                QuadLoop(current_element_x,
                         current_element_v,
                         i_thread,
                         e.quad_data_,
                         tmp,
                         tmp.backward_residual_);

                for (int k{}; k < e.n_tdof_; ++k) {
                  *grad_data++ = (fd_forward_data[k] - fd_backward_data[k])
                                 * two_diff_step_inv;
                }
                with_respect_to = orig_wrt;
              }
            }

            // assemble residual
            QuadLoop(current_element_x,
                     current_element_v,
                     i_thread,
                     e.quad_data_,
                     tmp,
                     e.residual_view_);
          }
        };

    mimi::utils::NThreadExe(assemble_element_residual_and_maybe_grad,
                            element_vectors_->size(),
                            n_threads_);
  }

  /// just as base
  virtual void AssembleBoundaryResidual(const mfem::Vector& current_x,
                                        const mfem::Vector& current_v) {}
};

} // namespace mimi::integrators
