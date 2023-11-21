#pragma once

#include "mimi/integrators/nonlinear_base.hpp"

namespace mimi::integrators {

/// basic integrator of nonlinear solids
/// given current x coordinate (NOT displacement)
/// Computes F and passes it to material
class NonliearSolid : public NonlinearBase {

  virtual const std::string& Name() const { return name_; }

  /// This one needs / stores
  /// - basis(shape) function derivative (at reference)
  /// - reference to target jacobian weight (det)
  /// - target to reference jacobian (inverse of previous)
  /// - basis derivative at target
  virtual void Prepare(const int quadrature_order = -1) {
    MIMI_FUNC()

    // get numbers to decide loop size
    const int n_elements = precomputed_->meshes_[0]->NURBSext->GetNE();
    const int n_threads =
        precomputed_->meshes_
            .size(); // n meshes should equal to nthrea config at the beginning

    // get dim
    dim_ = precomputed_->meshes_[0]->Dimension();

    // allocate element vectors and matrices
    element_matrices_ =
        std::make_unique<mimi::utils::Data<mfem::DenseMatrix>>(n_elements);
    element_vectors_ =
        std::make_unique<mimi::utils::Data<mfem::Vector>>(n_elements);

    // DShape - you get derivatives per dim. This is at reference
    auto& d_shapes = precomputed_->matrices_["d_shapes"];
    d_shapes.resize(n_elements);

    // target DShape
    auto& target_d_shapes = precomputed_->matrices_["target_d_shapes"];
    target_d_shapes.resize(n_elements);

    // element target_to_reference weight
    auto& target_to_reference_weights =
        precomputed_->scalars_["target_to_reference_weights"];
    target_to_reference_weights.resize(n_elements);

    // element reference_to_target jacobian (inverse of target_to_reference
    // jacobian)
    auto& reference_to_target_jacobians =
        precomputed_->matrices_["reference_to_target_jacobians"];
    reference_to_target_jacobians.resize(n_elements);

    // extract element geometry type
    geometry_type_ = precomputed_->elements_[0]->GetGeomType();

    // quadrature orders
    quadrature_orders_.resize(n_elements);

    auto precompute_at_elements_and_quads = [&](const int el_begin,
                                                const int el_end,
                                                const int i_thread) {
      // thread's obj
      auto& int_rules = precomputed_->int_rules_[i_thread];

      // element loop
      for (int i{el_begin}, i < el_end, ++i) {
        // deref i-th element of
        // elements, eltrans
        // std::vectors of mfem::Vector and mfem::DenseMatrix
        const auto& i_el = precomputed_->elements_[i];
        const auto& i_el_trans =
            *precomputed->target_to_reference_element_trans_[i];
        auto& i_d_shapes = d_shapes[i];
        auto& i_target_d_shapes = target_d_shapes[i];
        auto& i_target_to_reference_weights = target_to_reference_weights[i];
        auto& i_reference_to_target_jacobians =
            reference_to_target_jacobians[i];

        // get quad order
        const int q_order = (quadrature_order < 0)
                                ? i_el->GetFE()->GetOrder() * 2 + 3
                                : quadrature_order;

        // save this quad order
        quadrature_orders_[i] = q_order;

        // get int rule
        const mfem::IntegrationRule& ir =
            int_rules.Get(i_el->GetGeomType(), q_order);
        // prepare quad loop
        const int n_quad = ir.GetNPoints();
        const int n_dof = i_el->GetDof();

        // now, allocate space
        i_d_shapes.resize(n_quad);
        i_target_d_shapes.resize(n_quad);
        i_target_to_reference_weights.resize(n_quad);
        i_reference_to_target_jacobians.resize(n_quad);

        // also allocate element based vectors and matrix (assembly output)
        element_vectors_->operator[](i).SetSize(n_dof * dim_);
        element_matrices_->operator[](i).SetSize(n_dof * dim_, n_dof * dim_);

        for (int j{}; j < n_quad; ++j) {
          // get int point - this is just look up within ir
          const mfem::IntegrationPoint& ip = ir.IntPoint(j);
          i_el_trans.SetIntPoint(&ip);

          // get d shapes
          mfem::DenseMatrix& j_d_shape = i_shapes[j];
          mfem::DenseMatrix& j_target_d_shape = i_target_d_shapes[j];
          mfem::DenseMatrix& j_reference_to_target_jacobian =
              i_reference_to_target_jacobians[j];
          j_d_shape.SetSize(n_dof, dim_);
          j_target_d_shape.SetSize(n_dof, dim_);
          j_reference_to_target_jacobian.SetSize(dim_, dim_);

          //  Calc
          i_el->CalcDShape(ip, j_d_shape);
          mfem::CalcInverse(i_el_trans.Jacobian(),
                            j_reference_to_target_jacobian);
          mfem::Mult(j_d_shape,
                     j_reference_to_target_jacobian,
                     j_target_d_shape);

          // at last, trans weight
          i_target_to_reference_weights[j] = i_el_trans.Weight();
        }
      }
    };

    mimi::utils::NThreadExe(precompute_at_elements_and_quads,
                            n_elements,
                            n_threads);
  }

  virtual void AssembleDomainResidual(const mfem::Vector& current_x) {
    MIMI_FUNC()

    // get related precomputed values
    // d shapes are n_elem * n_quad (n_dof, n_dim)
    const auto& d_shapes = precomputed_->matrices["d_shapes"];
    const auto& target_d_shapes = precomputed_->matrices_["target_d_shapes"];
    // weights are n_elem * (n_quad)
    auto& target_to_reference_weights =
        precomputed_->scalars_["target_to_reference_weights"];
    // jacobians are n_elem * n_quad (n_dim, n_dim)
    auto& reference_to_target_jacobians =
        precomputed_->matrices_["reference_to_target_jacobians"];

    // lambda for nthread assemble
    auto assemble_element_residual = [&](const int begin,
                                         const int end,
                                         const end i_thread) {
      // for thread safety, each thread gets a thread-unsafe objects
      auto& int_rules = precomputed_->int_rules_[i_thread];

      // views / temporary copies
      mfem::DenseMatrix i_residual_mat_view; // matrix view of residual output
      mfem::Vector
          i_current_solution_vec; // contiguous copy of current solution
      mfem::DenseMatrix
          i_current_solution_mat_view; // matrix view of current solution

      for (int i{begin}; i < end; ++i) {
        const auto& i_vdof = precomputed_->v_dofs_[i];
        const auto& i_d_shapes = d_shapes[i];
        const auto& i_target_d_shapes = target_d_shapes[i];
        const auto& i_target_to_reference_weights =
            target_to_reference_weights[i];
        const auto& i_reference_to_target_jacobians =
            reference_to_target_jacobians[i];

        auto& i_residual = Base_::element_vectors_->operator[](i);
        i_residual = 0.0;

        // sizes
        const int n_dof = i_b_el->GetDof();

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
          const auto& q_target_d_shape = i_target_d_shapes[q];
        }
      }
    };
  }

  virtual void AssembleDomainGrad(const mfem::Vector& current_x) { MIMI_FUNC() }
};

} // namespace mimi::integrators
