#pragma once

#include <cmath>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/coefficients/nearest_distance.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

/// simple rewrite of mfem::AddMult_a_VWt()
template<typename DataType>
void AddMult_a_VWt(const DataType a,
                   const DataType* v_begin,
                   const DataType* v_end,
                   const DataType* w_begin,
                   const DataType* w_end,
                   DataType* aVWt) {
  DataType* out{aVWt};
  for (const DataType* wi{w_begin}; wi != w_end; ++wi) {
    const DataType aw = *wi * a;
    for (const DataType* vi{v_begin}; vi != v_end;) {
      (*out++) += aw * (*vi++);
    }
  }
}

/// implements methods presented in "Sauer and De Lorenzis. An unbiased
/// computational contact formulation for 3D friction (DOI: 10.1002/nme.4794)
class PenaltyContact : public NonlinearBase {
  constexpr static const int kMaxTrueDof = 50;
  constexpr static const int kDimDim = 9;

public:
  using Base_ = NonlinearBase;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;

  /// precomputed data at quad points
  struct QuadData {
    bool active_{false}; // needed for

    double integration_weight_;
    double det_dX_dxi_;
    double lagrange_;     // lambda_k
    double new_lagrange_; // lambda_k+1
    double penalty_;      // penalty factor
    double g_;            // normal gap

    mfem::Vector N_; // shape
    /// thanks to Jac's hint, it turns out we can just work with this for
    /// boundary
    mfem::DenseMatrix dN_dxi_; // don't really need to save this
    // mfem::DenseMatrix dN_dX_;  // used to compute F
    // mfem::DenseMatrix dxi_dX_; // J_target_to_reference

    mimi::coefficients::NearestDistanceBase::Query distance_query_;
    mimi::coefficients::NearestDistanceBase::Results distance_results_;
  };

  /// everything here is based on boundary element.
  /// consider that each variable has "boundary_" prefix
  struct BoundaryElementData {
    /// we only work with marked boundaries, so we keep two ids
    /// for convenience. One that belongs to marked boundary list (id_)
    /// and one that belongs to NBE(true_id_)
    int id_;
    int true_id_;
    int quadrature_order_;
    int geometry_type_;
    int n_quad_;
    int n_dof_; // this is not true dof
    int n_tdof_;

    std::shared_ptr<mfem::Array<int>> v_dofs_;
    mfem::DenseMatrix residual_view_; // we always assemble at the same place
    mfem::DenseMatrix grad_view_;     // we always assemble at the same place

    Vector_<QuadData> quad_data_;

    /// pointer to element and eltrans. don't need it,
    /// but maybe for further processing or something
    std::shared_ptr<mfem::NURBSFiniteElement> element_;
    std::shared_ptr<mfem::FaceElementTransformations> element_trans_;

    /// pointers to corresponding Base_::element_vectors_
    mfem::Vector* element_residual_;
    /// and Base_::element_matrices_
    mfem::DenseMatrix* element_grad_;

    const mfem::IntegrationRule&
    GetIntRule(mfem::IntegrationRules& thread_int_rules) const {
      MIMI_FUNC()

      return thread_int_rules.Get(geometry_type_, quadrature_order_);
    }
  };
  /// temporary containers required in element assembly
  /// mfem performs some fancy checks for allocating memories.
  /// So we create one for each thread
  struct TemporaryData {
    /// wraps element_state_data_
    mfem::Vector element_x_;
    /// wraps element_x_
    mfem::DenseMatrix element_x_mat_;
    /// wraps F_data
    mfem::DenseMatrix F_;
    /// wraps F_inv_data
    mfem::DenseMatrix F_inv_;
    /// wraps forward_residual data
    mfem::DenseMatrix forward_residual_;
    /// wraps backward residual data
    mfem::DenseMatrix backward_residual_;

    /// @brief
    /// @param element_x_data
    /// @param stress_data
    /// @param dN_dx_data
    /// @param F_data
    /// @param F_inv_data
    /// @param dim
    void SetData(double* element_x_data,
                 double* F_data,
                 double* F_inv_data,
                 double* forward_residual_data,
                 double* backward_residual_data,
                 const int para_dim,
                 const int dim) {
      MIMI_FUNC()

      element_x_.SetDataAndSize(element_x_data, kMaxTrueDof);
      element_x_mat_.UseExternalData(element_x_data, kMaxTrueDof, 1);
      // F_.UseExternalData(F_data, dim, para_dim);
      // F_inv_.UseExternalData(F_inv_data, dim, para_dim);
      F_.UseExternalData(F_data, dim, para_dim);
      F_inv_.UseExternalData(F_inv_data, para_dim, dim);
      forward_residual_.UseExternalData(forward_residual_data, kMaxTrueDof, 1);
      backward_residual_.UseExternalData(backward_residual_data,
                                         kMaxTrueDof,
                                         1);
    }

    void SetShape(const int n_dof, const int dim) {
      MIMI_FUNC()
      element_x_mat_.SetSize(n_dof, dim);
      forward_residual_.SetSize(n_dof, dim);
      backward_residual_.SetSize(n_dof, dim);
    }

    mfem::DenseMatrix&
    CurrentElementSolutionCopy(const mfem::Vector& current_all,
                               const BoundaryElementData& elem_data) {
      MIMI_FUNC()

      current_all.GetSubVector(*elem_data.v_dofs_, element_x_);
      return element_x_mat_;
    }
  };

protected:
  /// scene
  std::shared_ptr<mimi::coefficients::NearestDistanceBase>
      nearest_distance_coeff_ = nullptr;

  /// convenient constants - space dim (dim_) is in base
  int boundary_para_dim_;

  int n_threads_;

  int n_marked_boundaries_;

  Vector_<BoundaryElementData> boundary_element_data_;

  /// residual contribution indices
  /// can be used to copy al_lambda_
  /// size == n_marked_boundaries_;
  Vector_<int> marked_boundary_v_dofs_;

  /// we can update augmented lagrange vectors just by copying converged force
  /// https://hal.science/hal-01005280/document
  /// size == n_marked_boundaries_;
  mfem::Vector al_lambda_;

public:
  PenaltyContact(
      const std::shared_ptr<mimi::coefficients::NearestDistanceBase>&
          nearest_distance_coeff,
      const std::string& name,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : NonlinearBase(name, precomputed),
        nearest_distance_coeff_(nearest_distance_coeff) {}

  /// this one needs
  /// - shape function
  /// - weights of target to refrence
  virtual void Prepare(const int quadrature_order = -1) {
    MIMI_FUNC()

    // get numbers to decide loop size
    const int n_boundary_elements =
        precomputed_->meshes_[0]->NURBSext->GetNBE();
    n_threads_ =
        precomputed_->meshes_.size(); // size of mesh is nthread from Setup()

    // get dimension
    dim_ = precomputed_->meshes_[0]->Dimension();
    boundary_para_dim_ = dim_ - 1;

    // get marked boundary elements: this reduces size of the loop
    if (!boundary_marker_) {
      mimi::utils::PrintAndThrowError(Name(),
                                      "does not have a boundary marker.");
    }

    // prepare loop - deref and get a mesh to ask
    auto& mesh = *precomputed_->meshes_[0];
    auto& b_marker = *boundary_marker_;
    Base_::marked_boundary_elements_.reserve(n_boundary_elements);

    // loop boundary elements
    for (int i{}; i < n_boundary_elements; ++i) {
      auto& b_el = precomputed_->boundary_elements_[i];
      const int bdr_attr = mesh.GetBdrAttribute(i);
      if (b_marker[bdr_attr - 1] == 0) {
        continue;
      }
      Base_::marked_boundary_elements_.push_back(i);
    }
    Base_::marked_boundary_elements_.shrink_to_fit();

    // this is actually number of boundaries that contributes to assembly
    n_marked_boundaries_ = Base_::marked_boundary_elements_.size();

    // we will only keep marked boundary dofs
    // this means that we need a dof map
    al_lambda_.SetSize(n_marked_boundaries_ * dim_);

    // extract boundary geometry type
    boundary_geometry_type_ =
        precomputed_->boundary_elements_[0]->GetGeomType();

    // allocate element vectors and matrices
    Base_::boundary_element_matrices_ =
        std::make_unique<mimi::utils::Data<mfem::DenseMatrix>>();
    Base_::boundary_element_matrices_->Reallocate(n_marked_boundaries_);
    Base_::boundary_element_vectors_ =
        std::make_unique<mimi::utils::Data<mfem::Vector>>();
    Base_::boundary_element_vectors_->Reallocate(n_marked_boundaries_);

    boundary_element_data_.resize(n_marked_boundaries_);

    // now, weight of jacobian.
    auto precompute_at_elem_and_quad = [&](const int marked_b_el_begin,
                                           const int marked_b_el_end,
                                           const int i_thread) {
      // thread's obj
      auto& int_rules = precomputed_->int_rules_[i_thread];

      for (int i{marked_b_el_begin}; i < marked_b_el_end; ++i) {
        // we only need makred boundaries
        const int m = Base_::marked_boundary_elements_[i];

        // get boundary element data - note index i, not m!
        auto& i_bed = boundary_element_data_[i];
        i_bed.id_ = i;
        i_bed.true_id_ = m;

        // save (shared) pointers from global numbering
        i_bed.element_ = precomputed_->boundary_elements_[m];
        i_bed.geometry_type_ = i_bed.element_->GetGeomType();
        i_bed.element_trans_ =
            precomputed_->reference_to_target_boundary_trans_[m];
        i_bed.n_dof_ = i_bed.element_->GetDof();
        i_bed.n_tdof_ = i_bed.n_dof_ * dim_;
        i_bed.v_dofs_ = precomputed_->boundary_v_dofs_[m];

        // quick size check
        const int n_tdof = i_bed.n_tdof_;
        if (kMaxTrueDof < n_tdof) {
          mimi::utils::PrintAndThrowError(
              "kMaxTrueDof smaller than required space.",
              "Please recompile after setting a bigger number");
        }

        // now, setup some more from local properties
        i_bed.element_residual_ = &(*boundary_element_vectors_)[i];
        i_bed.element_residual_->SetSize(n_tdof);
        i_bed.residual_view_.UseExternalData(i_bed.element_residual_->GetData(),
                                             i_bed.n_dof_,
                                             dim_);

        i_bed.element_grad_ = &(*boundary_element_matrices_)[i];
        i_bed.element_grad_->SetSize(n_tdof, n_tdof);
        i_bed.grad_view_.UseExternalData(i_bed.element_grad_->GetData(),
                                         n_tdof,
                                         n_tdof);

        // let's prepare quad loop
        i_bed.quadrature_order_ = (quadrature_order < 0)
                                      ? i_bed.element_->GetOrder() * 2 + 3
                                      : quadrature_order;

        const mfem::IntegrationRule& ir = i_bed.GetIntRule(int_rules);
        i_bed.n_quad_ = ir.GetNPoints();
        i_bed.quad_data_.resize(i_bed.n_quad_);

        // this needs to come from volume element!
        for (int j{}; j < i_bed.n_quad_; ++j) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(j);
          i_bed.element_trans_->SetIntPoint(&ip);

          // first allocate, then start filling values
          auto& q_data = i_bed.quad_data_[j];
          q_data.integration_weight_ = ip.weight;
          q_data.N_.SetSize(i_bed.n_dof_);

          // /// These need to come from volume element!
          // q_data.dN_dX_.SetSize(i_bed.n_dof_, dim_);
          // q_data.dxi_dX_.SetSize(dim_, dim_);
          q_data.dN_dxi_.SetSize(i_bed.n_dof_, boundary_para_dim_);

          q_data.distance_results_.SetSize(boundary_para_dim_, dim_);
          q_data.distance_query_.SetSize(boundary_para_dim_);
          q_data.distance_query_.max_iterations_ = 20;

          // precompute
          // shape comes from boundary element
          i_bed.element_->CalcShape(ip, q_data.N_);
          i_bed.element_->CalcDShape(ip, q_data.dN_dxi_);
          // DShape should come from volume
          // get volume element
          // auto& i_el =
          // precomputed_->elements_[i_bed.element_trans_->Elem1No]; auto&
          // i_eltrans = i_bed.element_trans_->Elem1;
          // i_bed.element_trans_->Loc1.Transform(ip, eip);
          // std::cout << "dshape\n";
          // i_el->CalcDShape(eip, dN_dxi);
          // std::cout << "inv\n";
          // mfem::CalcInverse(i_eltrans->Jacobian(), q_data.dxi_dX_);
          // std::cout << "mult\n";
          // mfem::Mult(dN_dxi, q_data.dxi_dX_, q_data.dN_dX_);

          // q_data.det_dX_dxi_ = i_bed.element_trans_->Weight();

          // let's not forget to initialize values
          q_data.active_ = false;
          q_data.lagrange_ = 0.0;
          q_data.new_lagrange_ = 0.0;
          q_data.penalty_ = nearest_distance_coeff_->coefficient_;
          q_data.g_ = 0.0;
        }
      }
    };

    mimi::utils::NThreadExe(precompute_at_elem_and_quad,
                            n_marked_boundaries_,
                            n_threads_);

    // reserve enough space - supports can overlap so, be generous initially
    marked_boundary_v_dofs_.clear();
    marked_boundary_v_dofs_.reserve(
        n_marked_boundaries_ * dim_
        * std::pow((precomputed_->fe_spaces_[0]->GetMaxElementOrder() + 1),
                   boundary_para_dim_));
    for (const BoundaryElementData& bed : boundary_element_data_) {
      for (const int& vdof : *bed.v_dofs_) {
        marked_boundary_v_dofs_.push_back(vdof);
      }
    }

    // sort and get unique
    std::sort(marked_boundary_v_dofs_.begin(), marked_boundary_v_dofs_.end());
    auto last = std::unique(marked_boundary_v_dofs_.begin(),
                            marked_boundary_v_dofs_.end());
    marked_boundary_v_dofs_.erase(last, marked_boundary_v_dofs_.end());
    marked_boundary_v_dofs_.shrink_to_fit();
  }

  virtual void UpdateLagrange() {
    MIMI_FUNC();

    for (auto& be : boundary_element_data_) {
      for (auto& qd : be.quad_data_) {
        qd.lagrange_ = qd.new_lagrange_;
      }
    }
  }

  virtual void FillLagrange(const double value) {
    MIMI_FUNC()

    for (auto& be : boundary_element_data_) {
      for (auto& qd : be.quad_data_) {
        qd.lagrange_ = value;
      }
    }
  }

  void QuadLoop(const mfem::DenseMatrix& x,
                const int i_thread,
                Vector_<QuadData>& q_data,
                TemporaryData& tmp,
                mfem::DenseMatrix& residual_matrix) {
    MIMI_FUNC()
    double t_n_data[3];
    mimi::utils::Data<double> t_n(t_n_data, dim_);

    double* residual_begin = residual_matrix.GetData();
    for (QuadData& q : q_data) {
      // get current position and F
      x.MultTranspose(q.N_, q.distance_query_.query_.data());
      mfem::MultAtB(x, q.dN_dxi_, tmp.F_);

      // nearest distance query
      q.active_ = false; // init
      nearest_distance_coeff_->NearestDistance(q.distance_query_,
                                               q.distance_results_);
      q.distance_results_.ComputeNormal<true>(); // unit normal
      q.g_ = q.distance_results_.NormalGap();
      // activity check - only done if the following criteria is not met
      // if (!(q.g_ < q.lagrange_ / q.penalty_)) {
      if (!(q.lagrange_ < 0.0)) {
        // normalgap validity and angle tolerance
        constexpr const double angle_tol = 1.e-5;
        if (q.g_ > 0.
            || std::acos(
                   std::min(1., std::abs(q.g_) / q.distance_results_.distance_))
                   > angle_tol) {
          continue;
        }
      }
      const double p = q.lagrange_ + (q.penalty_ * q.g_);

      // this is from SIMO & LAURSEN (1990)
      // where they use Macauley bracket.
      // since we bring this residual directly to lhs,
      // sign for us is just flipped
      // https://doi.org/10.1016/0045-7949(92)90540-G
      if (!(p < 0.0)) {
        // q.active_ = false;
        continue;
      }
      // we made it til here.
      // if state is not frozen, we can save this as new lagrange
      if (!frozen_state_) {
        q.active_ = true;
        q.new_lagrange_ = p;
      }

      t_n.MultiplyAssign(q.new_lagrange_, q.distance_results_.normal_.data());

      // again, note no negative sign.
      AddMult_a_VWt(q.integration_weight_ * tmp.F_.Weight(),
                    q.N_.begin(),
                    q.N_.end(),
                    t_n.begin(),
                    t_n.end(),
                    residual_begin);
    }
  }

  virtual void AssembleBoundaryResidual(const mfem::Vector& current_x) {
    MIMI_FUNC()

    auto assemble_face_residual_and_maybe_grad =
        [&](const int begin, const int end, const int i_thread) {
          TemporaryData tmp;
          // create some space in stack
          double element_x_data[kMaxTrueDof];
          double F_data[kDimDim];
          double F_inv_data[kDimDim];
          double fd_forward_data[kMaxTrueDof];
          double fd_backward_data[kMaxTrueDof];
          tmp.SetData(element_x_data,
                      F_data,
                      F_inv_data,
                      fd_forward_data,
                      fd_backward_data,
                      boundary_para_dim_,
                      dim_);

          // this loops marked boundary elements
          for (int i{begin}; i < end; ++i) {
            // get bed
            BoundaryElementData& bed = boundary_element_data_[i];

            // reset residual
            bed.residual_view_ = 0.0;

            // set shape for tmp data
            tmp.SetShape(bed.n_dof_, dim_);
            mfem::DenseMatrix& current_element_x =
                tmp.CurrentElementSolutionCopy(current_x, bed);

            if (assemble_grad_) {
              assert(frozen_state_);
              double* grad_data = bed.grad_view_.GetData();
              double* solution_data = current_element_x.GetData();
              for (int j{}; j < bed.n_tdof_; ++j) {
                tmp.forward_residual_ = 0.0;
                tmp.backward_residual_ = 0.0;

                double& with_respect_to = *solution_data++;
                const double orig_wrt = with_respect_to;
                const double diff_step = std::abs(orig_wrt) * 1.0e-8;
                const double two_diff_step_inv = 1. / (2.0 * diff_step);

                with_respect_to = orig_wrt + diff_step;
                QuadLoop(current_element_x,
                         i_thread,
                         bed.quad_data_,
                         tmp,
                         tmp.forward_residual_);

                with_respect_to = orig_wrt - diff_step;
                QuadLoop(current_element_x,
                         i_thread,
                         bed.quad_data_,
                         tmp,
                         tmp.backward_residual_);

                for (int k{}; k < bed.n_tdof_; ++k) {
                  *grad_data++ = (fd_forward_data[k] - fd_backward_data[k])
                                 * two_diff_step_inv;
                }
                with_respect_to = orig_wrt;
              }
            }

            // assemble residual
            QuadLoop(current_element_x,
                     i_thread,
                     bed.quad_data_,
                     tmp,
                     bed.residual_view_);
          } // marked elem loop
        };

    mimi::utils::NThreadExe(assemble_face_residual_and_maybe_grad,
                            n_marked_boundaries_,
                            n_threads_);
  }

  virtual void AssembleBoundaryGrad(const mfem::Vector& current_x) {
    MIMI_FUNC()
  }

  /// boundaries are kept
  virtual void
  AddToGlobalBoundaryResidual(mfem::Vector& global_residual) const {
    MIMI_FUNC()
    for (const BoundaryElementData& bed : boundary_element_data_) {
      global_residual.AddElementVector(*bed.v_dofs_, *bed.element_residual_);
    }
  }

  virtual void AddToGlobalBoundaryGrad(mfem::SparseMatrix& global_grad) const {
    MIMI_FUNC()
    for (const BoundaryElementData& bed : boundary_element_data_) {
      global_grad.AddSubMatrix(*bed.v_dofs_,
                               *bed.v_dofs_,
                               *bed.element_grad_,
                               0);
    }
  }
};

} // namespace mimi::integrators
