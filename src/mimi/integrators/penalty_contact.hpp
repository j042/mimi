#pragma once

#include <cmath>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/coefficients/nearest_distance.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

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
    double g_{0.0};       // normal gap

    mfem::Vector N_; // shape
    // mfem::DenseMatrix dN_dxi_; // don't really need to save this
    mfem::DenseMatrix dN_dX_;  // used to compute F
    mfem::DenseMatrix dxi_dX_; // J_target_to_reference

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
    std::shared_ptr<mfem::IsoparametricTransformation> element_trans_;

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
    /// wraps stress_data
    mfem::DenseMatrix stress_;
    /// wraps dN_dx_data
    mfem::DenseMatrix dN_dx_;
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
                 double* stress_data,
                 double* dN_dx_data,
                 double* F_data,
                 double* F_inv_data,
                 double* forward_residual_data,
                 double* backward_residual_data,
                 const int dim) {
      MIMI_FUNC()

      element_x_.SetDataAndSize(element_x_data, kMaxTrueDof);
      element_x_mat_.UseExternalData(element_x_data, kMaxTrueDof, 1);
      stress_.UseExternalData(stress_data, dim, dim);
      dN_dx_.UseExternalData(dN_dx_data, kMaxTrueDof, 1);
      F_.UseExternalData(F_data, dim, dim);
      F_inv_.UseExternalData(F_inv_data, dim, dim);
      forward_residual_.UseExternalData(forward_residual_data, kMaxTrueDof, 1);
      backward_residual_.UseExternalData(backward_residual_data,
                                         kMaxTrueDof,
                                         1);
    }

    void SetShape(const int n_dof, const int dim) {
      MIMI_FUNC()

      element_x_mat_.SetSize(n_dof, dim);
      dN_dx_.SetSize(n_dof, dim);
      forward_residual_.SetSize(n_dof, dim);
      backward_residual_.SetSize(n_dof, dim);
    }

    mfem::DenseMatrix&
    CurrentElementSolutionCopy(const mfem::Vector& current_all,
                               const ElementData& elem_data) {
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
    lambda_al_.SetSize(n_marked_boundaries_ * dim_);

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

      mfem::DenseMatrix dN_dxi;

      for (int i{marked_b_el_begin}; i < marked_b_el_end; ++i) {
        // we only need makred boundaries
        const int m = Base_::marked_boundary_elements_[i];

        // get boundary element data
        auto& i_bed = boundary_element_data_[m];

        i_bed.id_ = i;
        i_bed.true_id_ = m;
        // save (shared) pointers from global numbering
        i_bed.element_ = precomputed_->boundary_elements_[m];
        i_bed.geometry_type_ = i_bed.element_->GetGeomType();
        i_bed.element_trans_ =
            precomputed_->reference_to_target_element_trans_[m];
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
                                             i_bed.n_dof,
                                             dim_);
        i_bed.element_grad_ = &(*element_matrices_)[i];
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
        dN_dxi.SetSize(i_bed.n_dof_, boundary_para_dim_);
        for (int j{}; j < i_el_data.n_quad_; ++j) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(j);
          i_bed.element_trans_->SetIntPoint(&ip);

          // first allocate, then start filling values
          auto& q_data = i_bed.quad_data_[j];
          q_data.integration_weight_ = ip.weight;
          q_data.N_.SetSize(i_bed.n_dof_);
          q_data.dN_dX_.SetSize(i_bed.n_dof_, boundary_para_dim_);
          q_data.dxi_dX_.SetSize(i_bed.n_dof_, boundary_para_dim_);
          q_data.distance_results_.SetSize(boundary_para_dim_, dim_);
          q_data.distance_query_.SetSize(boundary_para_dim_);

          // precompute
          i_bed.element_->CalcShape(ip, q_data.dN_);
          i_bed.element_->CalcDShape(ip, dN_dxi);
          mfem::CalcInverse(i_bed.element_trans_->Jacobian(), q_data.dxi_dX_);
          mfem::Mult(dN_dxi, q_data.dxi_dX_, q_data.dN_dX_);
          q_data.det_dX_dxi_ = i_bed.element_trans_->Weight();

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
      for (const int& vdof : bed.v_dofs_) {
        marked_boundary_v_dofs_.push_back(vdof);
      }
    }

    // sort and get unique
    std::sort(marked_boundary_v_dofs_.begin(), marked_boundary_v_dofs_.end());
    auto last = std::unique(marked_boundary_v_dofs_.begin(),
                            marked_boundary_v_dofs_.end());
    marked_boundary_v_dofs_.erase(last, marked_boundary_v_dofs_.end());
    marked_boundary_v_dofs_.shrinked_to_fit();
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

    for (QuadData& q : q_data) {
      // get current position and F
      x.MultTranspose(q.N_, q.distance_query_.query_.data());
      mfem::MultAtB(x, q.dN_dX_, tmp.F_);

      // nearest distance query
      q.active_ = false; // init
      nearest_distance_coeff_->NearestDistance(query, q_result);
      q_result.ComputeNormal<true>(); // unit normal
      q.g_ = q_result.NormalGap();

      // activity check
      // if lagrange value is set, we should compute this regardless of g_ value
      if (q.langrange_ < 0.0) {
      }
    }
  }

  virtual void AssembleBoundaryResidual(const mfem::Vector& current_x) {
    MIMI_FUNC()

    auto assemble_face_residual = [&](const int begin,
                                      const int end,
                                      const int i_thread) {
      // prepare per thread data - query, int_rule
      mimi::coefficients::NearestDistanceBase::Query query{};
      query.max_iterations_ = 20;
      query.query_.Reallocate(dim_); // allocate

      auto& int_rules = precomputed_->int_rules_[i_thread];

      // temp arrays
      mimi::utils::Data<double> traction_n(dim_);

      // element dependent densemat
      // this is always the same for each boundary of a single patch
      // we use densemat, becaue this has Weight() function implemented
      // which corresponds to sqrt(det(derivatives))
      mfem::DenseMatrix current_quad_derivatives(dim_, boundary_para_dim_);

      // convenient view of residual as matrix
      mfem::DenseMatrix i_residual_view;

      // this loops marked boundary elements
      for (int i{begin}; i < end; ++i) {
        // get this loop's objects
        // important to extract makred value.

        const int& i_mbe = Base_::marked_boundary_elements_[i];
        const auto& i_b_vdof = precomputed_->boundary_v_dofs_[i_mbe];
        auto& i_b_el = precomputed_->boundary_elements_[i_mbe];
        const auto& i_shapes = boundary_shapes[i_mbe];
        const auto& i_d_shapes = boundary_d_shapes[i_mbe];
        const auto& i_reference_to_target_weights =
            boundary_reference_to_target_weights[i_mbe];
        auto& i_results = nearest_distance_results_[i_mbe];
        const auto& i_lagranges = augmented_lagrange_multipliers[i_mbe];
        auto& i_new_lagranges = new_augmented_lagrange_multipliers[i_mbe];
        auto& i_normal_gaps = normal_gaps[i_mbe];
        auto& i_residual = Base_::boundary_element_vectors_->operator[](i_mbe);
        i_residual = 0.0;

        // sizes
        const int n_dof = i_b_el->GetDof();

        // setup helper matrices
        current_quad_derivatives.SetSize(dim_, boundary_para_dim_);
        i_residual_view.UseExternalData(i_residual.GetData(), n_dof, dim_);

        assert(dim_ == i_results[0].dim_);
        assert(boundary_para_dim_ == i_b_el->GetDim());

        // copy current solution
        mfem::Vector i_current_solution_vec;
        current_x.GetSubVector(*i_b_vdof, i_current_solution_vec);
        mfem::DenseMatrix i_current_solution(i_current_solution_vec.GetData(),
                                             n_dof,
                                             dim_);

        // prepare quad loop
        const auto& int_rule =
            int_rules.Get(boundary_geometry_type_,
                          boundary_quadrature_orders_[i_mbe]);

        // quad loop
        for (int q{}; q < int_rule.GetNPoints(); ++q) {

          const mfem::IntegrationPoint& ip = int_rule.IntPoint(q);
          const auto& q_shape = i_shapes[q];
          const auto& q_d_shape = i_d_shapes[q];
          const auto& q_reference_to_target_weight =
              i_reference_to_target_weights[q];
          const auto& q_lagrange = i_lagranges[q];
          auto& q_new_lagrange = i_new_lagranges[q];
          auto& q_result = i_results[q];
          auto& q_normal_gap = i_normal_gaps[q];

          // mark this result inactive first so that we don't have to set
          // this at each early exit check
          q_result.active_ = false;

          // form query - get current position
          i_current_solution.MultTranspose(q_shape, query.query_.data());
          // evaluate derivatives at current quad position
          mfem::MultAtB(i_current_solution,
                        q_d_shape,
                        current_quad_derivatives);

          // query nearest distance
          nearest_distance_coeff_->NearestDistance(query, q_result);

          // get normal gap - this also saves normal at this point
          q_normal_gap = NormalGap(q_result);

          // active checks
          //
          // 1. exact zero check. we put hold on this one
          // if (q_result.distance_ < nearest_distance_coeff_->tolerance_)

          // 2. normal gap orientation - we can probably merge condition (1)
          // here
          if (q_normal_gap > 0.) {
            mimi::utils::PrintDebug(
                "exiting quad loop because normal gap is positive");
            continue;
          }

          // 3. angle check
          constexpr const double angle_tolerance = 1.0e-5;
          if (std::acos(
                  std::min(1., std::abs(q_normal_gap) / q_result.distance_))
              > angle_tolerance) {
            mimi::utils::PrintDebug(
                "exiting quad loop because angle difference is too big");
            continue;
          }

          // set true to active
          q_result.active_ = true;

          // get traction - t_factor should come out as a negative value
          //
          // technically in literatures, t_factor should be positive
          // and we integrate negative traction.
          // for here, we just keep it as it is, since the results are the same
          //
          // lambda_new = penalty_factor * normal_distance + lambda_old
          q_new_lagrange =
              nearest_distance_coeff_->coefficient_ * q_normal_gap + q_lagrange;
          assert(q_new_lagrange < 0.0);
          for (int j{}; j < dim_; ++j) {
            traction_n[j] = q_new_lagrange * q_result.normal_[j];
          }

          // get sqrt of det of metric tensor
          q_result.query_metric_tensor_weight_ =
              current_quad_derivatives.Weight();

          // get product of all the weights
          // I don't think we need this(* q_reference_to_target_weight)
          const double weight =
              ip.weight * q_result.query_metric_tensor_weight_;

          // set residual
          for (int j{}; j < dim_; ++j) {
            for (int k{}; k < n_dof; ++k) {
              i_residual_view(k, j) += q_shape[k] * traction_n[j] * weight;
            }
          } // res update

        } // quad loop

      } // marked elem loop
    };

    mimi::utils::NThreadExe(assemble_face_residual,
                            marked_boundary_elements_.size(),
                            precomputed_->meshes_.size());
  }

  /// -eps [I - c_ab * tangent_a dyadic tangent_b]
  virtual void AssembleBoundaryGrad(const mfem::Vector& current_x) {
    MIMI_FUNC()

    // get related precomputed values
    auto& boundary_shapes = precomputed_->vectors_["boundary_shapes"];
    auto& boundary_d_shapes = precomputed_->matrices_["boundary_d_shapes"];
    // jacobian weights
    auto& boundary_reference_to_target_weights =
        precomputed_->scalars_["boundary_reference_to_target_weights"];
    // normal gaps
    auto& normal_gaps = precomputed_->scalars_["normal_gaps"];

    auto assemble_face_grad = [&](const int begin,
                                  const int end,
                                  const int i_thread) {
      auto& int_rules = precomputed_->int_rules_[i_thread];

      // thread local matrices
      // lack of better term, we call it cp_mat as mentione in literature
      // consists of metric tensor and curvature tensor.
      mfem::DenseMatrix cp_mat(boundary_para_dim_, boundary_para_dim_);
      // dTn_dx
      mfem::DenseMatrix dtn_dx(dim_, dim_);
      // outer product of first ders temporary
      mfem::DenseMatrix der1_vvt(dim_, dim_);
      // wraper for der1 to use in vvt
      mfem::Vector der1_wrap(nullptr, dim_);

      for (int i{begin}; i < end; ++i) {
        // get this loop's objects
        const int& i_mbe = Base_::marked_boundary_elements_[i];
        const auto& i_b_vdof = precomputed_->boundary_v_dofs_[i_mbe];
        auto& i_b_el = precomputed_->boundary_elements_[i_mbe];
        const auto& i_shapes = boundary_shapes[i_mbe];
        const auto& i_d_shapes = boundary_d_shapes[i_mbe];
        auto& i_results = nearest_distance_results_[i_mbe];
        auto& i_normal_gaps = normal_gaps[i_mbe];

        // matrix to assemble
        auto& i_grad = Base_::boundary_element_matrices_->operator[](i_mbe);
        i_grad = 0.0; // initialize

        // sizes
        const int n_dof = i_b_el->GetDof();

        // prepare quad loop
        const auto& int_rule =
            int_rules.Get(boundary_geometry_type_,
                          boundary_quadrature_orders_[i_mbe]);

        // quad loop
        for (int q{}; q < int_rule.GetNPoints(); ++q) {
          auto& q_result = i_results[q];
          if (!q_result.active_) {
            continue;
          }

          const mfem::IntegrationPoint& ip = int_rule.IntPoint(q);
          const auto& q_shape = i_shapes[q];
          const auto& q_d_shape = i_d_shapes[q];
          auto& q_normal_gap = i_normal_gaps[q];

          // create some shortcuts from results
          const auto& der1 = q_result.first_derivatives_;
          const auto& der2 = q_result.second_derivatives_;
          const auto& metric_tensor_weight =
              q_result.query_metric_tensor_weight_;
          const auto& normal = q_result.normal_;
          const double& penalty = nearest_distance_coeff_->coefficient_;

          // calc cp_mat - this is symmetric so has a potential to be reduced
          // for 3D for 2D, won't make any difference.
          for (int p_1{}; p_1 < boundary_para_dim_; ++p_1) {
            for (int p_2{}; p_2 < boundary_para_dim_; ++p_2) {
              auto& cp_mat_p_1_2 = cp_mat(p_1, p_2);

              for (int d{}; d < dim_; ++d) {
                // TODO make sure normal gap has correct sign
                cp_mat_p_1_2 += der1(p_1, d) * der1(p_2, d)
                                - q_normal_gap * normal[d] * der2(p_1, p_2, d);
              }
            }
          }
          // finalize by inverting this matrix
          cp_mat.Invert();

          // normal traction der
          // start with -eps * I, this will set everything else to zero
          // also sysmetric(, I think)
          dtn_dx.Diag(-penalty, dim_);
          // penalty * cp_mat_a_b * outer(der1, der1)
          for (int d_1{}; d_1 < dim_; ++d_1) {
            for (int d_2{}; d_2 < dim_; ++d_2) {
              auto& dtn_dx_d_1_2 = dtn_dx(d_1, d_2);

              for (int p_1{}; p_1 < boundary_para_dim_; ++p_1) {
                for (int p_2{}; p_2 < boundary_para_dim_; ++p_2) {
                  dtn_dx_d_1_2 += penalty * cp_mat(p_1, p_2) * der1(p_1, d_1)
                                  * der1(p_2, d_2);
                }
              }
            }
          } // dtn_dx

          const double weight = ip.weight * metric_tensor_weight;

          // the four for loops
          // let's at least try to have a contiguous access for one of it
          // isn't this also symmetric?
          //
          // trying to do the following
          // i_grad(d_1 * n_dof + f_1, d_2 * n_dof + f_2) += -q_shape(f_1) *
          // q_shape(f_2) * dtn_dx(d_1, d_2) * weight;
          double* grad_entry = i_grad.Data();
          for (int d_2{}; d_2 < dim_; ++d_2) {
            for (int f_2{}; f_2 < n_dof; ++f_2) {
              const auto& shape_2 =
                  q_shape[f_2]; // shape2 is same for this column

              for (int d_1{}; d_1 < dim_; ++d_1) {
                const auto& dtn_dx_d_1_2 =
                    dtn_dx(d_1, d_2); // same value applied for same dim

                for (int f_1{}; f_1 < n_dof; ++f_1) {
                  *grad_entry++ +=
                      -q_shape(f_1) * shape_2 * dtn_dx_d_1_2 * weight;
                }
              }
            }

          } // grad

        } // quad

      } // element
    };

    mimi::utils::NThreadExe(assemble_face_grad,
                            marked_boundary_elements_.size(),
                            precomputed_->meshes_.size());
  }

  virtual void AssembleFaceVector(
      const mfem::FiniteElement& element,
      const mfem::FiniteElement&,                   /* unused second element*/
      mfem::FaceElementTransformations& face_trans, /* target to reference */
      const mfem::Vector& current_solution,
      mfem::Vector& residual) {
    MIMI_FUNC()

    const int n_dim = element.GetDim();
    const int n_dof = element.GetDof();

    residual.SetSize(n_dim * n_dof);
    residual = 0.;

    mimi::utils::PrintInfo("ndof",
                           n_dof,
                           "n_dim",
                           n_dim,
                           "not doing anything!");
  }

  virtual void AssembleFaceGrad(
      const mfem::FiniteElement& element,
      const mfem::FiniteElement& /* not used */,
      mfem::FaceElementTransformations& face_trans /* target to reference */,
      const mfem::Vector& input_state /* current solution */,
      mfem::DenseMatrix& jac_res /* stiffness mat */) {

    MIMI_FUNC()

    const int n_dim = element.GetDim();
    const int n_dof = element.GetDof();

    jac_res.SetSize(n_dim * n_dof);
    jac_res = 0.;

    mimi::utils::PrintInfo("ndof",
                           n_dof,
                           "n_dim",
                           n_dim,
                           "not doing anything!");
  }
};

} // namespace mimi::integrators
