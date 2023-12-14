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
  using Vector_ = Base_::Vector<T>;
  using QuadratureMatrices_ = Base_::QuadratureMatrices_;
  using QuadratureScalars_ = Base_::QuadratureScalars_;

public:
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

  virtual void AssembleElementResidual(const mfem::Vector& input_element_x,
                                       const int& i_elem,
                                       const int& i_thread,
                                       mfem::Vector& output_element_residual) {
    MIMI_FUNC()

    // deref thread local
    auto& int_rules = precomputed_->int_rules_[i_thread];
    const auto& int_rule =
        int_rules.Get(geometry_type_, quadrature_orders_[i_elem]);
    auto& tmp = thread_local_temporaries_[i_thread];

    // deref look ups
    const auto& i_el = precomputed_->elements_[i_elem];
    // const auto& i_dN_dxi = precomputed_dN_dxi_->operator[](i_elem);
    const auto& i_dN_dX = precomputed_dN_dX_->operator[](i_elem);
    const auto& i_det_J_reference_to_target =
        precomputed_det_J_reference_to_target_->operator[](i_elem);
    const auto& i_J_target_to_reference =
        precomputed_J_target_to_reference_->operator[](i_elem);

    // deref to-saves
    auto& i_F = current_F_->operator[](i_elem);
    auto& i_F_inv = current_F_inv_->operator[](i_elem);
    auto& i_det_F = current_det_F_->operator[](i_elem);

    // material states
    auto& i_material_states = material_states_[i_elem];

    // let's assemble
    // 0.0 init output res
    output_element_residual = 0.0;
    // sizes
    const int n_dof = i_el->GetDof();
    // setup views in tmps
    tmp.element_vector_matrix_view_.UseExternalData(input_element_x.GetData(),
                                                    n_dof,
                                                    dim_);
    tmp.residual_matrix_view_.UseExternalData(output_element_residual.GetData(),
                                              n_dof,
                                              dim_);

    // quad loop
    for (int q{}; q < int_rule.GetNPoints(); ++q) {
      const mfem::IntegrationPoint& ip = int_rule.IntPoint(q);
      const auto& q_dN_dX = i_dN_dX[q];
      const auto& q_det_J_reference_to_target = i_det_J_reference_to_target[q];
      const auto& q_J_target_to_reference = i_J_target_to_reference[q];
      auto& q_F = i_F[q];
      auto& q_material_state = i_material_states[q];

      // get dx_dX (=F)
      mfem::MultAtB(tmp.element_vector_matrix_view_, q_dN_dX, q_F);

      // check where this needs to be integrated
      if (material_->UsesCauchy()) {
        // get F^-1 and det(F)
        auto& q_F_inv = i_F[q];
        auto& q_det_F = i_det_F[q];
        mfem::CalcInverse(q_F, q_F_inv);
        q_det_F = q_F.Weight();

        material_->EvaluateCauchy(q_F, i_thread, q_material_state, tmp.stress_);

        tmp.dN_dx_.SetSize(n_dof, dim_);
        mfem::Mult(q_dN_dX, q_F_inv, tmp.dN_dx_);
        mfem::AddMult_a(q_det_F * ip.weight * q_det_J_reference_to_target,
                        tmp.dN_dx_,
                        tmp.stress_,
                        tmp.residual_matrix_view_);
      } else {
        // call PK1
        material_->EvaluatePK1(q_F, i_thread, q_material_state, tmp.stress_);
        mfem::AddMult_a_ABt(ip.weight * q_det_J_reference_to_target,
                            q_dN_dX,
                            tmp.stress_,
                            tmp.residual_matrix_view_);
      }
    }
  }

  virtual void AssembleDomainResidual(const mfem::Vector& current_x,
                                      const mfem::Vector& current_v) {
    MIMI_FUNC()

    // get related precomputed values
    // d shapes are n_elem * n_quad (n_dof, n_dim)
    const auto& d_shapes = precomputed_->matrices_["d_shapes"];
    const auto& target_d_shapes = precomputed_->matrices_["target_d_shapes"];
    // weights are n_elem * (n_quad)
    const auto& reference_to_target_weights =
        precomputed_->scalars_["reference_to_target_weights"];
    // jacobians are n_elem * n_quad (n_dim, n_dim)
    const auto& target_to_reference_jacobians =
        precomputed_->matrices_["target_to_reference_jacobians"];
    auto& deformation_gradients =
        precomputed_->matrices_["deformation_gradients"];

    // lambda for nthread assemble
    auto assemble_element_residual = [&](const int begin,
                                         const int end,
                                         const int i_thread) {
      // for thread safety, each thread gets a thread-unsafe objects
      auto& int_rules = precomputed_->int_rules_[i_thread];

      // views / temporary copies
      mfem::DenseMatrix i_residual_mat_view; // matrix view of residual output
      mfem::Vector
          i_current_solution_vec; // contiguous copy of current solution
      mfem::DenseMatrix
          i_current_solution_mat_view; // matrix view of current solution

      // temporary stress holder can be either PK1 or cauchy, based on material
      mfem::DenseMatrix stress(dim_, dim_);
      // gradients of basis in physical (current) configuration
      mfem::DenseMatrix dN_dx;

      for (int i{begin}; i < end; ++i) {
        // in
        const auto& i_el = precomputed_->elements_[i];
        const auto& i_vdof = precomputed_->v_dofs_[i];
        const auto& i_d_shapes = d_shapes[i];
        const auto& i_target_d_shapes = target_d_shapes[i];
        const auto& i_reference_to_target_weights =
            reference_to_target_weights[i];
        const auto& i_target_to_reference_jacobians =
            target_to_reference_jacobians[i];

        // out / local save
        auto& i_material_states = material_states_[i];
        auto& i_deformation_gradients = deformation_gradients[i];
        auto& i_deformation_gradient_inverses =
            deformation_gradient_inverses[i];
        auto& i_deformation_gradient_weights = deformation_gradient_weights[i];
        auto& i_residual = Base_::element_vectors_->operator[](i);
        i_residual = 0.0;

        // sizes
        const int n_dof = i_el->GetDof();

        // copy current solution
        current_x.GetSubVector(*i_vdof, i_current_solution_vec);
        // create views
        i_current_solution_mat_view.UseExternalData(
            i_current_solution_vec.GetData(),
            n_dof,
            dim_);
        i_residual_mat_view.UseExternalData(i_residual.GetData(), n_dof, dim_);

        // quad loop
        const auto& int_rule =
            int_rules.Get(geometry_type_, quadrature_orders_[i]);
        for (int q{}; q < int_rule.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = int_rule.IntPoint(q);
          const auto& q_target_d_shape = i_target_d_shapes[q]; // dN_dX
          const auto& q_reference_to_target_weight =
              i_reference_to_target_weights[q];
          auto& q_deformation_gradient = i_deformation_gradients[q];
          auto& q_material_state = i_material_states[q];

          // get dx_dX
          mfem::MultAtB(i_current_solution_mat_view,
                        q_target_d_shape,
                        q_deformation_gradient);

          // check where this needs to be integrated
          if (material_->UsesCauchy()) {
            dN_dx.SetSize(n_dof, dim_);
            // get F^-1 and det(F)
            auto& q_deformation_gradient_inverse =
                i_deformation_gradient_inverses[q];
            auto& q_deformation_gradient_weight =
                i_deformation_gradient_weights[q];
            mfem::CalcInverse(q_deformation_gradient,
                              q_deformation_gradient_inverse);
            q_deformation_gradient_weight = q_deformation_gradient.Weight();

            material_->EvaluateCauchy(q_deformation_gradient,
                                      i_thread,
                                      q_material_state,
                                      stress);

            mfem::Mult(q_target_d_shape, q_deformation_gradient_inverse, dN_dx);
            mfem::AddMult_a(q_deformation_gradient_weight * ip.weight
                                * q_reference_to_target_weight,
                            dN_dx,
                            stress,
                            i_residual_mat_view);
          } else {
            // call PK1
            material_->EvaluatePK1(q_deformation_gradient,
                                   i_thread,
                                   q_material_state,
                                   stress);
            mfem::AddMult_a_ABt(ip.weight * q_reference_to_target_weight,
                                q_target_d_shape,
                                stress,
                                i_residual_mat_view);
          }
        }
      }
    };

    mimi::utils::NThreadExe(assemble_element_residual,
                            element_vectors_->size(),
                            n_threads_);
  }

  virtual void AssembleDomainGrad(const mfem::Vector& current_x) {
    MIMI_FUNC()

    constexpr const double diff_step = 1.0e-8;
    constexpr const double two_diff_step = 2.0e-8;

    // currently we do FD. Maybe we will have AD, maybe manually.
    // central difference
    auto fd_grad = [&](const int start, const int end, const int i_thread) {
      mfem::Vector element_vector;
      mfem::Vector forward_element_residual;
      mfem::Vector backward_element_residual;
      for (int i{start}; i < end; ++i) {
        // copy element matrix
        current_x.GetSubVector(*precomputed_->v_dofs_[i], element_vector);

        // output matrix
        mfem::DenseMatrix& i_grad_mat = element_matrices_->operator[](i);

        const int n_dof = element_vector.Size();
        forward_element_residual.SetSize(n_dof);
        backward_element_residual.SetSize(n_dof);
        for (int j{}; j < n_dof; ++j) {
          double& with_respect_to = element_vector[j];
          const double orig_wrt = with_respect_to;
          // one step forward
          with_respect_to = orig_wrt + diff_step;
          AssembleElementResidual(element_vector,
                                  i,
                                  i_thread,
                                  forward_element_residual);
          // one step back
          with_respect_to = orig_wrt - diff_step;
          AssembleElementResidual(element_vector,
                                  i,
                                  i_thread,
                                  backward_element_residual);

          // (forward - backward) /  (2 * step)
          // with pointer
          double* grad_col = &i_grad_mat(0, j);
          const double* f_res = forward_element_residual.GetData();
          const double* b_res = backward_element_residual.GetData();
          const double* f_res_end = f_res + n_dof;
          for (; f_res != f_res_end;) {
            *grad_col++ = ((*f_res++) - (*b_res++)) / two_diff_step;
          }

          // reset with respect to
          with_respect_to = orig_wrt;
        }
      }
    };

    MaterialState::freeze_ = true;
    mimi::utils::NThreadExe(fd_grad, element_matrices_->size(), n_threads_);
    MaterialState::freeze_ = false;
  }
};

} // namespace mimi::integrators
