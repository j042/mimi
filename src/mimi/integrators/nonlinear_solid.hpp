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
  constexpr static const int kMaxTrueDof = 50;
  constexpr static const int kDimDim = 9;

public:
  using Base_ = NonlinearBase;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;
  using QuadratureMatrices_ = Vector_<Vector_<mfem::DenseMatrix>>;
  using QuadratureScalars_ = Vector_<Vector_<double>>;

  /// precomputed data at quad points
  struct QuadData {
    double integration_weight_;
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

    void SetData(double* element_state_data,
                 double* stress_data,
                 double* dN_dx_data,
                 double* F_data,
                 double* F_inv_data,
                 const int dim) {
      MIMI_FUNC()

      element_state_.SetData(element_state_data, kMaxTrueDof);
      element_state_view_.UseExternalData(element_state_data, kMaxTrueDof, 1);
      stress_.UseExternalData(stress_data, dim, dim);
      dN_dx_.UseExternalData(dN_dx_data, kMaxTrueDof, 1);
      F_.UseExternalData(F_data, dim, dim);
      F_inv_.UseExternalData(F_inv_data, dim, dim);
    }

    void SetShape(const int n_dof, const int dim) {
      MIMI_FUNC()

      element_state_view_.SetSize(n_dof, dim);
      dN_dx_.SetSize(n_dof, dim);
    }

    mfem::DenseMatrix&
    CurrentElementSolutionCopy(const mfem::Vector& current_all,
                               const ElementData& elem_data) {
      MIMI_FUNC()

      current_all.GetSubVector(*elem_data.v_dofs_, element_state_);
      return element_state_view_;
    }
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
        if (kMaxTrueDof < n_tdof) {
          mimi::utils::PrintAndThrowError(
              "kMaxTrueDof smaller than required space.",
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
          q_data.integration_weight_ = ip.Weight();

          q_data.dN_dxi_.SetSize(i_el_data.n_dof_, dim_);
          q_data.dN_dX_.SetSize(i_el_data.n_dof_, dim_);
          q_data.dxi_dX_.SetSize(dim_, dim_);
          q_data.material_state_ = material_->CreateState();

          i_el_data.element_->CalcDShape(ip, q_data.dN_dxi_);
          mfem::CalcInverse(i_el_data.element_trans_.Jacobian(),
                            q_data.dxi_dX_);
          mfem::Mult(q_data.dN_dxi_, q_data.dxi_dX_, q_data.dN_dX_);
          q_data.det_dX_dxi_ = i_el_data.element_trans_.Weight();
        }
      }
    };

    mimi::utils::NThreadExe(precompute_at_elements_and_quads,
                            n_elements,
                            n_threads_);
  }

  /// Performs quad loop with element data and temporary data
  void QuadLoop(const mfem::DenseMatrix& x,
                const int i_thread,
                MaterialBase& material,
                Vector_<QuadData>& q_data,
                TemporaryData& tmp,
                mfem::DenseMatrix& residual_matrix) {
    MIMI_FUNC()

    for (QuadData& q : q_data) {
      // get dx_dX = x * dN_dX
      mfem::MultAtB(x, q.dN_dX_, tmp.F_);

      // currently we will just use PK1
      material.EvaluatePK1(tmp.F_, i_thread, q.material_state_, tmp.stress_);
      mfem::AddMult_a_ABt(q.integration_weight_ * q.det_dX_dxi_,
                          q.dN_dX_,
                          tmp.stress_,
                          residual_matrix);

      // alternatively, this. But, this does not converge.
      // check what's wrong with this
      //
      // material_->EvaluateCauchy(tmp.F_,
      //                           i_thread,
      //                           q.material_state_,
      //                           tmp.stress_);
      // mfem::CalcInverse(tmp.F_, tmp.F_inv_);
      // mfem::Mult(q.dN_dX_, tmp.F_inv_, tmp.dN_dx_);
      // mfem::AddMult_a(tmp.F_.Det() * q.integration_weight_ * q.det_dX_dxi_,
      //                 tmp.dN_dx_,
      //                 tmp.stress_,
      //                 e.residual_view_);
    }
  }

  virtual void AssembleDomainResidual(const mfem::Vector& current_x) {
    MIMI_FUNC()

    // lambda for nthread assemble
    auto assemble_element_residual_and_maybe_grad =
        [&](const int begin, const int end, const int i_thread) {
          TemporaryData tmp;
          // create some space in stack
          double element_state_data[kMaxTrueDof];
          double stress_data[kDimDim];
          double dN_dx_data[kMaxTrueDof];
          double F_data[kDimDim];
          double F_inv_data[kDimDim];
          tmp.SetData(element_state_data,
                      stress_data,
                      dN_dx_data,
                      F_data,
                      F_inv_data,
                      dim_);

          for (int i{begin}; i < end; ++i) {
            // in
            e = element_data_[i];
            e.residual_view_ = 0.0;

            // set shape for tmp data
            tmp.SetShape(e.n_dof_, dim_);

            // get current element solution as matrix
            mfem::DenseMatrix& current_solution =
                tmp.CurrentElementSolutionCopy(current_x, e);

            QuadLoop(current_solution,
                     i_thread,
                     *material_,
                     e.quad_data_,
                     tmp,
                     e.residual_view);
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
