#pragma once

#include "mimi/integrators/materials.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/n_thread_exe.hpp"

namespace mimi::integrators {

/// basic integrator of nonlinear solids
/// given current x coordinate (NOT displacement)
/// Computes F and passes it to material
class NonliearSolid : public NonlinearBase {

public:
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;
  using QuadratureMatrices_ = Vector_<Vector_<mfem::DenseMatrix>>;
  using QuadratureScalars_ = Vector_<Vector_<double>>;

protected:
  /// temporary containers required in element assembly
  /// mfem performs some fancy checks for allocating memories.
  /// So we create one for each thread
  struct Temporary {
    mfem::DenseMatrix residual_matrix_view_;
    mfem::Vector element_vector_copies_;
    mfem::Vector element_vector_matrix_view_;
    mfem::DenseMatrix stress_;
    mfem::DenseMatrix dN_dx_;
  };

  Vector_<Temporary> thread_local_temporaries_;

public:
  /// material states (n_elements * n_quads)
  Vector_<Vector_<MaterialState>> material_states_;

  /// for smooth nthread exe with FD or AD, we need element assembly
  /// that we can call element wise. So here, we will save
  /// pointers to precomputed entities for direct access
  const QuadratureMatrices_* precomputed_dN_dxi_;
  const QuadratureMatrices_* precomputed_dN_dX_;
  const QuadratureScalars_* precomputed_det_J_reference_to_target_;
  const QuadratureMatrices_* precomputed_J_target_to_reference_;

  QuadratureMatrices_* current_F_;
  QuadratureMatrices_* current_F_inv_;
  QuadratureScalars_* current_det_F_;

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
        material_states_[i].resize(n_quad);

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

    // now, save pointers locally
    precomputed_dN_dxi_ = &d_shapes;
    precomputed_dN_dX_ = &target_d_shapes;
    precomputed_det_J_reference_to_target_ = &reference_to_target_weights;
    precomputed_J_target_to_reference_ = &target_to_reference_jacobians;

    current_F_ = &deformation_gradients;
    current_F_inv_ = &deformation_gradients_inverses;
    current_det_F_ = &deformation_gradient_weights;

    // thread safety committee
    thread_local_temporaries_.resize(n_threads);
    // allocate stress
    for (auto& tlt : thread_local_temporaries_) {
      tlt.stress_.SetSize(dim_, dim_);
    }
  }

  /// element level assembly.
  /// currently copy of AssemblyDomainResidual.
  /// meant to be used for FD
  /// or, AD, where we can assemble both LHS and RHS contributions
  virtual void
  AssembleElementResidual(const mfem::Vector& input_element_x,
                          const int& i_elem,
                          const int& i_thread,
                          mfem::Vector& output_element_residual) const {
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
    auto& i_F_inv = current_F_inv_->opeartor[](i_elem);
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
      auto& q_F_inv = i_F[q];
      auto& q_i_det_F = i_det_F[q];
      auto& q_material_state = i_material_states[q];
      // get dx_dX (=F)
      mfem::MultAtB(tmp.element_vector_matrix_view_, q_dN_dX, q_F);

      // get F^-1 and det(F)
      mfem::CalcInverse(q_F, q_F_inv);
      q_det_F = q_F.Weight();

      // evaluate cauchy or PK1 stress
      material_->EvaluateStress(q_material_state, q_F, tmp.stress_);

      // check where this needs to be integrated
      if (material_->PhysicalIntegration()) {
        // stress is probably cauchy
        tmp.dN_dx_.SetSize(n_dof, dim_);
        mfem::Mult(q_dN_dX, q_F_inv, tmp.dN_dx_);
        // scalar multiplcation to stress for integration
        tmp.stress_ *= q_det_F * ip.weight * q_det_J_reference_to_target;
        mfem::AddMultABt(tmp.dN_dx_, tmp.stress_, tmp.residual_matrix_view_);
      } else {
        // stress is probably PK1
        tmp.stress_ *= ip.weight * q_det_J_reference_to_target;
        mfem::AddMultABt(q_dN_dX, tmp.stress_, tmp.residual_matrix_view_);
      }
    }
  }

}

virtual void
AssembleDomainResidual(const mfem::Vector& current_x) {
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
  // maybe we won't save those
  auto& deformation_gradient_inverses =
      precomputed_->matrices_["deformation_gradient_inverses"];
  auto& deformation_gradient_weights =
      precomputed_->scalars_["deformation_gradient_weights"];

  // lambda for nthread assemble
  auto assemble_element_residual = [&](const int begin,
                                       const int end,
                                       const int i_thread) {
    // for thread safety, each thread gets a thread-unsafe objects
    auto& int_rules = precomputed_->int_rules_[i_thread];

    // views / temporary copies
    mfem::DenseMatrix i_residual_mat_view; // matrix view of residual output
    mfem::Vector i_current_solution_vec; // contiguous copy of current solution
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
      auto& i_deformation_gradients = deformation_gradients[i];
      auto& i_deformation_gradient_inverses = deformation_gradient_inverses[i];
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
        auto& q_deformation_gradient_inverse =
            i_deformation_gradient_inverses[q];
        auto& q_deformation_gradient_weight = i_deformation_gradient_weights[q];

        // get dx_dX
        mfem::MultAtB(i_current_solution_mat_view,
                      q_target_d_shape,
                      q_deformation_gradient);

        // we may not need this
        // also can do it in if(PhysicalIntegration())
        // TODO talk to Dr Z
        mfem::CalcInverse(q_deformation_gradient,
                          q_deformation_gradient_inverse);
        q_deformation_gradient_weight = q_deformation_gradient.Weight();

        // evaluate cauchy or PK1 stress
        material_->EvaluateStress(i_material_states[q],
                                  q_deformation_gradient,
                                  stress);

        // check where this needs to be integrated
        if (material_->PhysicalIntegration()) {
          // stress is probably cauchy
          dN_dx.SetSize(n_dof, dim_);
          mfem::Mult(q_target_d_shape, q_deformation_gradient_inverse, dN_dx);
          stress *= q_deformation_gradient_weight * ip.weight
                    * q_reference_to_target_weight;
          mfem::AddMultABt(dN_dx, stress, i_residual_mat_view);
        } else {
          // stress is probably PK1
          stress *= ip.weight * q_reference_to_target_weight;
          mfem::AddMultABt(q_target_d_shape, stress, i_residual_mat_view);
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

  // currently we do FD
  // I realize, we need a separate element assembly..
}
};

} // namespace mimi::integrators
