#pragma once

#include "mimi/integrators/materials.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"

namespace mimi::integrators {

/// basic integrator of nonlinear solids
/// given current x coordinate (NOT displacement)
/// Computes F and passes it to material
class NonliearSolid : public NonlinearBase {
public:
  /// material states (n_elements * n_quads)
  mimi::utils::Vector<mimi::utils::Vector<MaterialState>> material_states_;

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

    // element reference_to_target weight
    auto& reference_to_target_weights =
        precomputed_->scalars_["reference_to_target_weights"];
    reference_to_target_weights.resize(n_elements);

    // element target_to_reference jacobian (inverse of reference_to_target
    // jacobian)
    auto& target_to_reference_jacobians =
        precomputed_->matrices_["target_to_reference_jacobians"];
    target_to_reference_jacobians.resize(n_elements);

    // deformation gradient is saved from the latest residual calculation
    // we save this, as gradient call comes right after the first residual
    // calculation this is F = dx/dX
    auto& deformation_gradients =
        precomputed_->matrices_["deformation_gradients"];
    deformation_gradients.resize(n_elements);

    // deformation gradient inverse is also saved
    // TODO check if we really need to save this. For example, for Grad
    // F^-1
    // this is used in
    auto& deformation_gradients_inverses =
        precomputed_->matrices_["deformation_gradient_inverses"];
    deformation_gradients_inverses.resize(n_elements);

    // det(F)
    auto& deformation_gradient_weights =
        precomputed_->scalars_["deformation_gradient_weights"];
    deformation_gradient_weights.resize(n_elements);

    // extract element geometry type
    geometry_type_ = precomputed_->elements_[0]->GetGeomType();

    // quadrature orders - this is per elements
    quadrature_orders_.resize(n_elements);

    // material states - this is per elements per quad points
    material_states_.resize(n_elements);

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
            *precomputed->reference_to_target_element_trans_[i];
        auto& i_d_shapes = d_shapes[i];
        auto& i_target_d_shapes = target_d_shapes[i];
        auto& i_reference_to_target_weights = reference_to_target_weights[i];
        auto& i_target_to_reference_jacobians =
            target_to_reference_jacobians[i];
        auto& i_deformation_gradients = deformation_gradients[i];
        auto& i_deformation_gradient_inverses =
            deformation_gradient_inverses[i];
        auto& i_deformation_gradient_weights = deformation_gradient_weights[i];

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

        // allocate state. maybe we want to value initialize here.
        // default init for now (this is default_init_vector)
        material_states_[i]
            .resize(n_quad)

            // now, allocate space
            i_d_shapes.resize(n_quad);
        i_target_d_shapes.resize(n_quad);
        i_reference_to_target_weights.resize(n_quad);
        i_target_to_reference_jacobians.resize(n_quad);
        i_deformation_gradients.resize(n_quad);
        i_deformation_gradient_inverses.resize(n_quad);
        i_deformation_gradient_weights.resize(n_quad);

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
          mfem::DenseMatrix& j_target_to_reference_jacobian =
              i_target_to_reference_jacobians[j];
          mfem::DenseMatrix& j_deformation_gradient =
              i_deformation_gradients[j];
          mfem::DenseMatrix& j_deformation_gradient =
              i_deformation_gradient_inverses[j];
          j_d_shape.SetSize(n_dof, dim_);
          j_target_d_shape.SetSize(n_dof, dim_);
          j_target_to_reference_jacobian.SetSize(dim_, dim_);
          // here, we just allocate the matrix
          j_deformation_gradient.SetSize(dim_, dim_);
          j_deformation_gradient_inverse.SetSize(dim_, dim_);

          //  Calc
          i_el->CalcDShape(ip, j_d_shape);
          mfem::CalcInverse(i_el_trans.Jacobian(),
                            j_target_to_reference_jacobian);
          mfem::Mult(j_d_shape,                      // dN_dxi
                     j_target_to_reference_jacobian, // dxi_dX
                     j_target_d_shape);              // dN_dX

          // at last, trans weight
          i_reference_to_target_weights[j] = i_el_trans.Weight();
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
    const auto& reference_to_target_weights =
        precomputed_->scalars_["reference_to_target_weights"];
    // jacobians are n_elem * n_quad (n_dim, n_dim)
    const auto& target_to_reference_jacobians =
        precomputed_->matrices_["target_to_reference_jacobians"];
    auto& deformation_gradients =
        precomputed_->matrices_["deformation_gradients"];
    auto& deformation_gradient_inverses =
        precomputed_->matrices_["deformation_gradient_inverses"];
    auto& deformation_gradient_weights =
        precomputed_->scalars_["deformation_gradient_weights"];

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

      // temporary stress holder can be either PK1 or cauchy, based on material
      mfem::DenseMatrix stress(dim_, dim_);

      for (int i{begin}; i < end; ++i) {
        // in
        const auto& i_vdof = precomputed_->v_dofs_[i];
        const auto& i_d_shapes = d_shapes[i];
        const auto& i_target_d_shapes = target_d_shapes[i];
        const auto& i_reference_to_target_weights =
            reference_to_target_weights[i];
        const auto& i_target_to_reference_jacobians =
            target_to_reference_jacobians[i];

        // out
        auto& i_deformation_gradients = deformation_gradients[i];
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
          const auto& q_target_d_shape = i_target_d_shapes[q]; // dN_dX
          const auto& q_reference_to_target_weight =
              i_reference_to_target_weights[q];
          auto& q_deformation_gradient = i_deformation_gradients[q];

          // get dx_dX
          mfem::MultAtB(i_current_solution_mat_view,
                        q_target_d_shape,
                        q_deformation_gradient);

          // evaluate cauchy or PK1 stress
          material_->EvaluateStress(i_material_states[q],
                                    q_deformation_gradient,
                                    stress);

          // check where this needs to be integrated
          if (material_->PhysicalIntegration()) {
            // stress is probably cauchy
            auto&

          } else {
            // stress is probably PK1
            stress *= ip.weight * q_reference_to_target_weight;
            mfem::AddMultABt(q_target_d_shape, stress, i_residual_mat_view);
          }
        }
      }
    };
  }

  virtual void AssembleDomainGrad(const mfem::Vector& current_x) { MIMI_FUNC() }
};

} // namespace mimi::integrators
