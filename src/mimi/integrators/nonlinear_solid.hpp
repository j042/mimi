#pragma once

#include "mimi/integrators/materials.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/n_thread_exe.hpp"

namespace mimi::integrators {

/// basic integrator of nonlinear solids
/// given current x coordinate (NOT displacement)
/// Computes F and passes it to material
class NonlinearSolid : public NonlinearBase {

public:
  using Base_ = NonlinearBase;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;
  using QuadratureMatrices_ = Vector_<Vector_<mfem::DenseMatrix>>;
  using QuadratureScalars_ = Vector_<Vector_<double>>;

  /// precomputed data at quad points
  struct QuadData {
    mfem::DenseMatrix dN_dxi_; // don't really need to save this
    mfem::DenseMatrix dN_dX_;
    double det_dX_dxi_;
    mfem::DenseMatrix dxi_dX_; // J_target_to_reference
    std::shared_ptr<MaterialState> material_state_;
  }

  struct ElementData {
    int quadrature_order_;
    int geometry_type_;
    int n_quad_;
    int n_dof_; // this is not true dof

    std::shared_ptr<mfem::Array<int>> v_dofs_;
    mfem::DenseMatrix residual_view_; // we always assemble at the same place
    mfem::DenseMatrix grad_view_;     // we always assemble at the same place

    Vector_<QuaData> quad_data_;

    /// pointer to element and eltrans. don't need it,
    /// but maybe for further processing or something
    std::shared_ptr<mfem::NURBSFiniteElement> element_;
    std::shared_ptr<mfem::IsoparametricTransformation> element_trans_;

    /// pointers to corresponding Base_::element_vectors_
    mfem::Vector* element_residual_;
    /// and Base_::element_matrices_
    mfem::DenseMatrix* element_grad_;

    const mfem::IntegrationRule&
    GetIntRule(mfem::IntegrationRules& thread_int_rules) const {
      return thread_int_rules.Get(geometry_type, quadrature_order_);
    }
  }

  /// temporary containers required in element assembly
  /// mfem performs some fancy checks for allocating memories.
  /// So we create one for each thread
  struct TemporaryData {
    /// allocate some static space
    double F_data_[9];
    double F_inv_data_[9];
    double stress_data_[9];

    /// set max dof. maybe you will have to change
    constexpr static const int kMaxTrueDof = 50;
    double element_state_data_[kMaxTrueDof];
    double dN_dx_data_[kMaxTrueDof];

    /// wraps element_state_data_
    mfem::Vector element_state_;
    /// wraps element_state_
    mfem::DenseMatrix element_state_view_;
    /// wraps stress_data_
    mfem::DenseMatrix stress_;
    /// wraps dN_dx_data_
    mfem::DenseMatrix dN_dx_;
    /// wraps F_data_
    mfem::DenseMatrix F_;
    /// wraps F_inv_
    mfem::DenseMatrix F_inv_;
  };

protected:
  Vector_<TemporaryData> thread_local_temporaries_;

  int n_threads_;

  std::shared_ptr<MaterialBase> material_;

  Vector_<ElementData> element_data_;

public:
  NonlinearSolid(
      const std::string& name,
      const std::shared_ptr<MaterialBase>& material,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : NonlinearBase(name, precomputed),
        material_{material} {}

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
    // n meshes should equal to nthrea config at the beginning
    n_threads_ = precomputed_->meshes_.size();

    // get dim
    dim_ = precomputed_->meshes_[0]->Dimension();

    // setup material
    material_->Setup(dim_, n_threads_);

    // allocate element vectors and matrices
    element_matrices_ =
        std::make_unique<mimi::utils::Data<mfem::DenseMatrix>>(n_elements);
    element_vectors_ =
        std::make_unique<mimi::utils::Data<mfem::Vector>>(n_elements);

    // extract element geometry type
    geometry_type_ = precomputed_->elements_[0]->GetGeomType();

    // allocate element data
    element_data_.resize(n_elements);

    auto precompute_at_elements_and_quads = [&](const int el_begin,
                                                const int el_end,
                                                const int i_thread) {
      // thread's obj
      auto& int_rules = precomputed_->int_rules_[i_thread];

      // element loop
      for (int i{el_begin}; i < el_end; ++i) {
        // prepare element level data
        auto& i_el_data = element_data_[i];

        // save (shared) pointers to element and el_trans
        i_el_data.element_ = precomputed_->elements_[i];
        i_el_data.geometry_type_ = i_el_data.element_->GetGeomType();
        i_el_data.element_trans_ =
            precomputed_->reference_to_target_element_trans_[i];
        i_el_data.n_dof_ = i_el_data.element_->GetDof();
        i_el_data.v_dofs_ = precomputed_->v_dofs_[i];

        // check suitability of temp data
        // for structure simulations, tdof and vdof are the same
        const int n_tdof = i_el_data.n_dof_ * dim_;
        if (TemporaryData::kMaxTrueDof < n_tdof) {
          mimi::utils::PrintAndThrowError(
              "TemporaryData::kMaxTrueDof smaller than required space.",
              "Please recompile after setting a bigger number");
        }

        // also allocate element based vectors and matrix (assembly output)
        i_el_data.element_residual_ = &(*element_vectors_)[i];
        i_el_data.element_residual_->SetSize(n_tdof);
        i_el_data.residual_view_.UseExternalData(
            i_el_data.element_residual_->GetData(),
            i_el_data.n_dof_,
            dim_);
        i_el_data.element_grad_ = &(*element_matrices_)[i];
        i_el_data.element_grad_->SetSize(n_tdof, n_tdof);
        i_el_data.grad_view_.UseExternalData(i_el_data.element_grad_->GetData(),
                                             n_tdof);

        // get quad order
        i_el_data.quadrature_order_ =
            (quadrature_order < 0) ? i_el_data.element_->GetOrder() * 2 + 3
                                   : quadrature_order;

        // get int rule
        const mfem::IntegrationRule& ir = i_el_data.GetIntRule(int_rules);

        // prepare quad loop
        i_el_data.n_quad_ = ir.GetNPoints();
        quad_data_.resize(i_el_data.n_quad_);

        for (int j{}; j < n_quad; ++j) {
          // get int point - this is just look up within ir
          const mfem::IntegrationPoint& ip = ir.IntPoint(j);
          i_el_data.element_trans_.SetIntPoint(&ip);

          auto& q_data = i_el_data.quad_data_[j];

          q_data.dN_dxi_.SetSize(i_el_data.n_dof_, dim_);
          q_data.dN_dX_.SetSize(i_el_data.n_dof_, dim_);
          q_data.dxi_dX_.SetSize(dim_, dim_);
          q_data.material_state_ = material_->CreateState();

          i_el_data.element_->CalcDShape(ip, q_data.dN_dxi_);
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
                            n_threads_);

    // now, save pointers locally
    precomputed_dN_dxi_ = &d_shapes;
    precomputed_dN_dX_ = &target_d_shapes;
    precomputed_det_J_reference_to_target_ = &reference_to_target_weights;
    precomputed_J_target_to_reference_ = &target_to_reference_jacobians;

    current_F_ = &deformation_gradients;
    current_F_inv_ = &deformation_gradient_inverses;
    current_det_F_ = &deformation_gradient_weights;

    // thread safety committee
    thread_local_temporaries_.resize(n_threads_);
    // allocate stress
    for (auto& tlt : thread_local_temporaries_) {
      tlt.stress_.SetSize(dim_, dim_);
    }
  }

  /// element level assembly.
  /// currently copy of AssemblyDomainResidual.
  /// meant to be used for FD
  /// or, AD, where we can assemble both LHS and RHS contributions
  /// As we currently use for FD, we will turn off state accumulation
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

  virtual void AssembleDomainResidual(const mfem::Vector& current_x) {
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
