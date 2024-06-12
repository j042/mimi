#pragma once

#include <cmath>
#include <fstream>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/coefficients/nearest_distance.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

/// simple rewrite of mfem::AddMult_a_VWt()
template<typename DataType>
void Ptr_AddMult_a_VWt(const DataType a,
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

template<typename DerivativeContainer, typename NormalContainer>
static inline void
ComputeUnitNormal(const DerivativeContainer& first_derivatives,
                  NormalContainer& unit_normal) {
  MIMI_FUNC()

  const int dim = unit_normal.Size();
  const double* jac_d = first_derivatives.GetData();
  double* unit_normal_d = unit_normal.GetData();
  if (dim == 2) {
    const double d0 = jac_d[0];
    const double d1 = jac_d[1];
    const double inv_norm2 = 1. / std::sqrt(d0 * d0 + d1 * d1);
    unit_normal_d[0] = d1 * inv_norm2;
    unit_normal_d[1] = -d0 * inv_norm2;

    // this should be either 2d or 3d
  } else {
    const double d0 = jac_d[0];
    const double d1 = jac_d[1];
    const double d2 = jac_d[2];
    const double d3 = jac_d[3];
    const double d4 = jac_d[4];
    const double d5 = jac_d[5];

    const double n0 = d1 * d5 - d2 * d4;
    const double n1 = d2 * d3 - d0 * d5;
    const double n2 = d0 * d4 - d1 * d3;

    const double inv_norm2 = 1. / std::sqrt(n0 * n0 + n1 * n1 + n2 * n2);

    unit_normal_d[0] = n0 * inv_norm2;
    unit_normal_d[1] = n1 * inv_norm2;
    unit_normal_d[2] = n2 * inv_norm2;
  }
}

/// implements normal contact methods presented in "De Lorenzis.
// A large deformation frictional contact formulation using NURBS-based
// isogeometric analysis"
class AveragedPenaltyContact : public NonlinearBase {
public:
  using Base_ = NonlinearBase;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;

  // post process
  std::unique_ptr<mfem::SparseMatrix> m_mat_;
  mfem::CGSolver mass_inv_;
  mfem::DSmoother mass_inv_prec_;
  mfem::UMFPackSolver mass_inv_direct_;
  mfem::Vector mass_b_;        // for rhs
  mfem::Vector mass_x_;        // for answer
  mfem::Vector last_residual_; // Forces
  double last_area_;           // for p = forces / A
  mimi::utils::Data<double> last_force_;

  /// precomputed data at quad points
  struct QuadData {
    bool active_{false}; // needed for

    double integration_weight_;
    double det_dX_dxi_;

    double det_J_;
    double lagrange_;     // lambda_k
    double new_lagrange_; // lambda_k+1 - this is also last_p
    double g_;            // normal gap
    double old_g_;        // normal gap

    inline void Record(const bool really,
                       const double detj,
                       const double new_lag,
                       const double g,
                       const bool activ) {
      MIMI_FUNC()

      if (!really)
        return;
      det_J_ = detj;
      new_lagrange_ = new_lag;
      old_g_ = g_;
      g_ = g;
      active_ = activ;
    }

    mfem::Vector N_; // shape
    /// thanks to Jac's hint, it turns out we can just work with this for
    /// boundary
    mfem::DenseMatrix dN_dxi_; // shape grad
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

    // we keep activity info in element level.
    bool active_ = false;

    std::shared_ptr<mfem::Array<int>> v_dofs_;
    mfem::Array<int> dofs_;
    mfem::Array<int> local_v_dofs_;
    mfem::Array<int> local_dofs_;

    Vector_<QuadData> quad_data_;

    /// pointer to element and eltrans. don't need it,
    /// but maybe for further processing or something
    std::shared_ptr<mfem::NURBSFiniteElement> element_;
    std::shared_ptr<mfem::FaceElementTransformations> element_trans_;

    // direct access id to sparse matrix
    // as we only loop over marked boundaries, keep this pointer for easy access
    std::shared_ptr<mimi::utils::Vector<int>> sparse_matrix_A_ids_;

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
    mfem::Vector element_x_;
    mfem::DenseMatrix element_x_mat_;
    mfem::DenseMatrix J_;
    mfem::DenseMatrix local_residual_;
    mfem::DenseMatrix local_grad_;
    mfem::DenseMatrix forward_residual_;

    mfem::Vector element_gap_;

    mimi::coefficients::NearestDistanceBase::Query distance_query_;
    mimi::coefficients::NearestDistanceBase::Results distance_results_;

    mfem::Vector normal_;

    int dim_{-1};

    void SetDim(const int dim) {
      MIMI_FUNC()

      dim_ = dim;
      assert(dim_ > 0);

      distance_query_.SetSize(dim);
      distance_results_.SetSize(dim - 1, dim);
      J_.SetSize(dim_, dim_ - 1);
      normal_.SetSize(dim_);
    }

    void SetDof(const int n_dof) {
      MIMI_FUNC()

      assert(dim_ > 0);
      element_x_.SetSize(n_dof * dim_);
      element_x_mat_.UseExternalData(element_x_.GetData(), n_dof, dim_);
      local_residual_.SetSize(n_dof, dim_);
      local_grad_.SetSize(n_dof * dim_, n_dof * dim_);
      forward_residual_.SetSize(n_dof, dim_);
      element_gap_.SetSize(n_dof);
    }

    mfem::DenseMatrix&
    CurrentElementSolutionCopy(const mfem::Vector& current_all,
                               const BoundaryElementData& elem_data) {
      MIMI_FUNC()

      current_all.GetSubVector(*elem_data.v_dofs_, element_x_);
      return element_x_mat_;
    }

    mfem::Vector& CurrentElementGap(const mfem::Vector& gap_all,
                                    const BoundaryElementData& elem_data) {
      MIMI_FUNC()

      gap_all.GetSubVector(elem_data.local_dofs_, element_gap_);
      return element_gap_;
    }

    mfem::DenseMatrix& ComputeJ(const mfem::DenseMatrix& dndxi) {
      MIMI_FUNC()
      mfem::MultAtB(element_x_mat_, dndxi, J_);
      return J_;
    }

    void ComputeNearestDistanceQuery(const mfem::Vector& N) {
      MIMI_FUNC()
      element_x_mat_.MultTranspose(N, distance_query_.query_.data());
    }

    double Gap(const mfem::Vector& N) const {
      MIMI_FUNC()
      const double* g_d = element_gap_.GetData();
      const double* n_d = N.GetData();
      double gap{};
      for (int i{}; i < N.Size(); ++i) {
        gap += n_d[i] * g_d[i];
      }
      return gap;
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
  /// size == n_marked_boundaries_;
  Vector_<int> marked_boundary_v_dofs_;
  Vector_<int> local_marked_v_dofs_;
  Vector_<int> local_marked_dofs_;

  /// two vectors for load balancing
  /// first we visit all quad points and also mark quad activity
  Vector_<bool> element_activity_;
  Vector_<int> active_elements_;

  /// these are numerator and denominator
  mfem::Vector average_gap_;
  mfem::Vector area_;

  Vector_<TemporaryData> temporary_data_;

public:
  AveragedPenaltyContact(
      const std::shared_ptr<mimi::coefficients::NearestDistanceBase>&
          nearest_distance_coeff,
      const std::string& name,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : NonlinearBase(name, precomputed),
        nearest_distance_coeff_(nearest_distance_coeff) {}

  /// this one needs
  /// - shape function
  /// - weights of target to refrence
  virtual void Prepare() {
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

    // extract boundary geometry type
    boundary_geometry_type_ =
        precomputed_->boundary_elements_[0]->GetGeomType();

    boundary_element_data_.resize(n_marked_boundaries_);

    last_force_.Reallocate(dim_);

    const int default_q_order =
        RuntimeCommunication()->GetInteger("contact_quadrature_order", -1);

    temporary_data_.resize(n_threads_);

    auto precompute_at_elem_and_quad = [&](const int marked_b_el_begin,
                                           const int marked_b_el_end,
                                           const int i_thread) {
      // thread's obj
      auto& int_rules = precomputed_->int_rules_[i_thread];
      auto& tmp = temporary_data_[i_thread];
      tmp.SetDim(dim_);

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
        i_bed.dofs_.SetSize(i_bed.v_dofs_->Size() / dim_);
        for (int k{}, l{}; k < i_bed.v_dofs_->Size(); k += dim_, ++l) {
          i_bed.dofs_[l] = (*i_bed.v_dofs_)[k] / dim_;
        }

        // quick size check
        const int n_tdof = i_bed.n_tdof_;

        i_bed.sparse_matrix_A_ids_ = precomputed_->boundary_A_ids_[m];

        // let's prepare quad loop
        i_bed.quadrature_order_ = (default_q_order < 0)
                                      ? i_bed.element_->GetOrder() * 2 + 3
                                      : default_q_order;

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
          q_data.dN_dxi_.SetSize(i_bed.n_dof_, boundary_para_dim_);

          // precompute
          // shape comes from boundary element
          i_bed.element_->CalcShape(ip, q_data.N_);
          i_bed.element_->CalcDShape(ip, q_data.dN_dxi_);
          q_data.det_dX_dxi_ = i_bed.element_trans_->Weight();

          // let's not forget to initialize values
          q_data.active_ = false;
          q_data.lagrange_ = 0.0;
          q_data.new_lagrange_ = 0.0;
          q_data.g_ = 0.0;
          q_data.old_g_ = 0.0;
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

    // get last value -> this is the size we need for extracting
    // using marked_boundary_v_dofs_, we can extract values from global
    // vector
    // using local_marked_v_dof_, we can directly map global v_dof to local
    // v_dof
    // same holds for local marked dof, but this is just
    local_marked_v_dofs_.assign(*(--marked_boundary_v_dofs_.end()) + 1, -1);
    local_marked_dofs_.assign(local_marked_v_dofs_.size() / dim_, -1);
    int v_counter{}, counter{};
    for (const int m_vdof : marked_boundary_v_dofs_) {
      if (v_counter % dim_ == 0) {
        // marked_boundary_v_dofs_ is sorted, which means xyzxyzxyz...
        local_marked_dofs_[m_vdof / dim_] = counter++;
      }
      local_marked_v_dofs_[m_vdof] = v_counter++;
    }
    last_residual_.SetSize(local_marked_v_dofs_.size());

    // prepare local dofs for local, reduced length vectors
    for (auto& bed : boundary_element_data_) {
      // v_dofs are xxxxyyyyzzzz
      bed.local_v_dofs_.SetSize(bed.n_tdof_);
      bed.local_dofs_.SetSize(bed.n_dof_);
      for (int i{}; i < bed.n_tdof_; ++i) {
        if (i < bed.n_dof_) {
          bed.local_dofs_[i] = local_marked_dofs_[(*bed.v_dofs_)[i] / dim_];
        }
        bed.local_v_dofs_[i] = local_marked_v_dofs_[(*bed.v_dofs_)[i]];
      }
      mimi::utils::PrintInfo("dofs");
      bed.local_dofs_.Print();
      mimi::utils::PrintInfo("vdofs");
      bed.local_v_dofs_.Print();
    }

    // save indices
    if (RuntimeCommunication()->ShouldSave("contact_forces")) {
      RuntimeCommunication()->SaveVector("contact_residual_index_mapping",
                                         local_marked_v_dofs_.data(),
                                         local_marked_v_dofs_.size());
      RuntimeCommunication()->SaveVector("marked_boundary_v_dofs",
                                         marked_boundary_v_dofs_.data(),
                                         marked_boundary_v_dofs_.size());
    }

    // resize and init
    const int local_dof_size =
        *std::max_element(local_marked_dofs_.begin(), local_marked_dofs_.end())
        + 1;
    average_gap_.SetSize(local_dof_size);
    average_gap_ = 0.0;
    area_.SetSize(local_dof_size);
    area_ = 0.0;
  }

  // we will average lagrange
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
        qd.new_lagrange_ = value;
      }
    }
  }

  void InitializeAverageGapAndArea() {
    MIMI_FUNC()

    double* ag_d = average_gap_.GetData();
    double* ad = area_.GetData();
    for (int i{}; i < average_gap_.Size(); ++i) {
      ad[i] = 0.0;
      ag_d[i] = 0.0;
    }
  }

  void
  AverageGap(const mfem::Vector& current_x, const int nthreads, double& area) {

    InitializeAverageGapAndArea();

    std::mutex gap_mutex;
    // lambda for nthread assemble
    auto prepare_average_gap = [&](const int begin,
                                   const int end,
                                   const int i_thread) {
      double local_area{};
      auto& tmp = temporary_data_[i_thread];
      for (int i{begin}; i < end; ++i) {
        BoundaryElementData& bed = boundary_element_data_[i];
        tmp.SetDof(bed.n_dof_);

        // get current element solution as matrix
        mfem::DenseMatrix& current_element_x =
            tmp.CurrentElementSolutionCopy(current_x, bed);

        bool any_active = false;
        const double penalty = nearest_distance_coeff_->coefficient_;
        const int* ids = bed.local_dofs_.GetData();
        for (QuadData& q : bed.quad_data_) {
          tmp.ComputeNearestDistanceQuery(q.N_);
          nearest_distance_coeff_->NearestDistance(tmp.distance_query_,
                                                   tmp.distance_results_);
          tmp.distance_results_.ComputeNormal<true>(); // unit normal
          const double g = tmp.distance_results_.NormalGap();

          mfem::DenseMatrix& J = tmp.ComputeJ(q.dN_dxi_);
          const double fac = q.integration_weight_ * J.Weight();
          local_area += fac;

          if (!(g < 0.)) { // tried going through all, but didn't work so well.
                           // TODO Try again
            continue;
          }

          // push right away
          double* a = area_.GetData();
          double* avg_g = average_gap_.GetData();
          const double* shape = q.N_.GetData();
          const double g_fac = fac * g;
          // real push
          std::lock_guard<std::mutex> lock(gap_mutex);
          for (int k{}; k < bed.n_dof_; ++k) {
            const double N = shape[k];
            a[ids[k]] += N * fac;
            avg_g[ids[k]] += N * g_fac;
          }
        }
      }
      // area contribution
      std::lock_guard<std::mutex> lock(gap_mutex);
      area += local_area;
    };

    mimi::utils::NThreadExe(prepare_average_gap,
                            n_marked_boundaries_,
                            (nthreads < 0) ? n_threads_ : nthreads);

    // divide
    const double* a_d = area_.GetData();
    double* ag_d = average_gap_.GetData();
    for (int i{}; i < average_gap_.Size(); ++i) {
      const double a = a_d[i];
      if (a != 0.0) {
        ag_d[i] /= a;
      }
    }
  }

  bool
  QuadLoop(Vector_<QuadData>& q_data,
           TemporaryData& tmp,
           mfem::DenseMatrix& residual,
           const bool keep_record,
           mimi::utils::Data<double>& force /* only if applicable*/) const {
    MIMI_FUNC()
    bool any_active = false;
    const double penalty = nearest_distance_coeff_->coefficient_;

    for (QuadData& q : q_data) {
      const double g = tmp.Gap(q.N_);
      assert(std::isfinite(g));

      if (!(g < 0.)) {
        continue;
      }

      // we need derivatives to get normal.
      mfem::DenseMatrix& J = tmp.ComputeJ(q.dN_dxi_);
      const double det_J = J.Weight();
      ComputeUnitNormal(J, tmp.normal_);

      // // check for angle and sign of normal gap for the first augmentation
      // // loop otherwise, we detemine exit criteria based on p value.
      // if (!(q.lagrange_ < 0.0)) {
      //   // normalgap validity and angle tolerance
      //   // angle between two vectors:  acos(a * b / (a.norm * b.norm))
      //   // difference * unit_normal is g, distance is denom.
      //   constexpr const double angle_tol = 1.e-5;
      //   if (g > 0.
      //       || std::acos(
      //              std::min(1., std::abs(g) /
      //              tmp.distance_results_.distance_)) > angle_tol) {
      //     q.Record(keep_record, det_J, 0.0, g, false);
      //     continue;
      //   }
      // }

      // we allow reduction of p, as long as it doesn't change sign
      // implementation here has flipped sign so we cut at 0.0
      // see https://doi.org/10.1016/0045-7949(92)90540-G
      const double p = std::min(q.lagrange_ + penalty * g, 0.0);
      if (p == 0.0) {
        // this quad no longer contributes to contact enforcement
        q.Record(keep_record, det_J, 0.0, g, false);
        continue;
      }

      // maybe only a minor difference, but we don't want to keep records from
      // line search or FD computations
      q.Record(keep_record, det_J, p, g, true);

      // set active after checking p
      any_active = true;

      // this one has negative sign as we use the normal from gauss point
      const double fac = q.integration_weight_ * det_J * p;
      Ptr_AddMult_a_VWt(-fac,
                        q.N_.begin(),
                        q.N_.end(),
                        tmp.normal_.begin(),
                        tmp.normal_.end(),
                        residual.GetData());
      if (keep_record) {
        // no negative sign, because we want to keep the sign to point inwards
        force.Add(fac, tmp.normal_.begin());
      }
    }

    return any_active;
  }

  /// rhs (for us, we have them on lhs - thus fliped sign) - used in line
  /// search
  virtual void AddBoundaryResidual(const mfem::Vector& current_x,
                                   const int nthreads,
                                   mfem::Vector& residual) {
    MIMI_FUNC()
    double dummy_area{};
    AverageGap(current_x, nthreads, dummy_area);

    std::mutex residual_mutex;

    auto assemble_face_residual_and_maybe_grad =
        [&](const int begin, const int end, const int i_thread) {
          auto& tmp = temporary_data_[i_thread];
          double area_dummy;
          mimi::utils::Data<double> force_dummy;
          // this loops marked boundary elements
          for (int i{begin}; i < end; ++i) {
            // get bed
            BoundaryElementData& bed = boundary_element_data_[i];
            tmp.SetDof(bed.n_dof_);
            tmp.local_residual_ = 0.0;

            mfem::DenseMatrix& current_element_x =
                tmp.CurrentElementSolutionCopy(current_x, bed);
            tmp.CurrentElementGap(average_gap_, bed);

            // assemble residual
            // this function is called either at the last step at "melted"
            // staged or during line search, keep_record = False
            const bool any_active = QuadLoop(bed.quad_data_,
                                             tmp,
                                             tmp.local_residual_,
                                             false,
                                             force_dummy);

            // only push if any of quad point was active
            if (any_active) {
              const std::lock_guard<std::mutex> lock(residual_mutex);
              residual.AddElementVector(*bed.v_dofs_,
                                        tmp.local_residual_.GetData());
            }
          } // marked elem loop
        };

    mimi::utils::NThreadExe(assemble_face_residual_and_maybe_grad,
                            n_marked_boundaries_,
                            (nthreads < 0) ? n_threads_ : nthreads);
  }

  virtual void AddBoundaryGrad(const mfem::Vector& current_x,
                               const int nthreads,
                               mfem::SparseMatrix& grad) {
    MIMI_FUNC()
    double dummy_area{};
    AverageGap(current_x, nthreads, dummy_area);

    std::mutex residual_mutex;
    // lambda for nthread assemble
    auto assemble_boundary_residual_and_grad_then_contribute =
        [&](const int begin, const int end, const int i_thread) {
          auto& tmp = temporary_data_[i_thread];
          double area_dummy;
          mimi::utils::Data<double> force_dummy;
          for (int i{begin}; i < end; ++i) {
            BoundaryElementData& bed = boundary_element_data_[i];
            tmp.SetDof(bed.n_dof_);
            tmp.local_residual_ = 0.0;

            // get current element solution as matrix
            mfem::DenseMatrix& current_element_x =
                tmp.CurrentElementSolutionCopy(current_x, bed);
            tmp.CurrentElementGap(average_gap_, bed);

            // assemble residual
            const bool any_active = QuadLoop(bed.quad_data_,
                                             tmp,
                                             tmp.local_residual_,
                                             false,
                                             force_dummy);

            // skip if nothing's active
            if (!any_active) {
              continue;
            }

            double* grad_data = tmp.local_grad_.GetData();
            double* solution_data = current_element_x.GetData();
            const double* residual_data = tmp.local_residual_.GetData();
            const double* fd_forward_data = tmp.forward_residual_.GetData();
            for (int j{}; j < bed.n_tdof_; ++j) {
              tmp.forward_residual_ = 0.0;

              double& with_respect_to = *solution_data++;
              const double orig_wrt = with_respect_to;
              const double diff_step = std::abs(orig_wrt) * 1.0e-8;
              const double diff_step_inv = 1. / diff_step;

              with_respect_to = orig_wrt + diff_step;
              QuadLoop(bed.quad_data_,
                       tmp,
                       tmp.forward_residual_,
                       false,
                       force_dummy);

              for (int k{}; k < bed.n_tdof_; ++k) {
                *grad_data++ =
                    (fd_forward_data[k] - residual_data[k]) * diff_step_inv;
              }
              with_respect_to = orig_wrt;
            }

            // push right away
            std::lock_guard<std::mutex> lock(residual_mutex);
            double* A = grad.GetData();
            const double* local_A = tmp.local_grad_.GetData();
            const auto& A_ids = *bed.sparse_matrix_A_ids_;
            for (int k{}; k < A_ids.size(); ++k) {
              A[A_ids[k]] += *local_A++;
            }
          }
        };

    mimi::utils::NThreadExe(assemble_boundary_residual_and_grad_then_contribute,
                            n_marked_boundaries_,
                            (nthreads < 0) ? n_threads_ : nthreads);
  };

  /// assembles residual and grad at the same time. this should also have
  /// assembled the last converged residual
  virtual void AddBoundaryResidualAndGrad(const mfem::Vector& current_x,
                                          const int nthreads,
                                          const double grad_factor,
                                          mfem::Vector& residual,
                                          mfem::SparseMatrix& grad) {
    last_area_ = 0.0;
    AverageGap(current_x, nthreads, last_area_);

    last_force_.Fill(0.0);
    const bool save_residual =
        RuntimeCommunication()->ShouldSave("contact_forces");
    if (save_residual) {
      last_residual_ = 0.0;
    }
    std::mutex residual_mutex;
    // lambda for nthread assemble
    auto assemble_boundary_residual_and_grad_then_contribute =
        [&](const int begin, const int end, const int i_thread) {
          auto& tmp = temporary_data_[i_thread];
          mimi::utils::Data<double> tl_force(dim_);
          for (int i{begin}; i < end; ++i) {
            BoundaryElementData& bed = boundary_element_data_[i];
            tmp.SetDof(bed.n_dof_);
            tmp.local_residual_ = 0.0;

            // get current element solution as matrix
            mfem::DenseMatrix& current_element_x =
                tmp.CurrentElementSolutionCopy(current_x, bed);
            tmp.CurrentElementGap(average_gap_, bed);

            // assemble residual
            const bool any_active = QuadLoop(bed.quad_data_,
                                             tmp,
                                             tmp.local_residual_,
                                             true,
                                             tl_force);

            // skip if nothing's active
            if (!any_active) {
              continue;
            }

            double* grad_data = tmp.local_grad_.GetData();
            double* solution_data = current_element_x.GetData();
            const double* residual_data = tmp.local_residual_.GetData();
            const double* fd_forward_data = tmp.forward_residual_.GetData();
            for (int j{}; j < bed.n_tdof_; ++j) {
              tmp.forward_residual_ = 0.0;

              double& with_respect_to = *solution_data++;
              const double orig_wrt = with_respect_to;
              const double diff_step = std::abs(orig_wrt) * 1.0e-8;
              const double diff_step_inv = 1. / diff_step;

              with_respect_to = orig_wrt + diff_step;
              QuadLoop(bed.quad_data_,
                       tmp,
                       tmp.forward_residual_,
                       false,
                       tl_force /* this either */);

              for (int k{}; k < bed.n_tdof_; ++k) {
                *grad_data++ =
                    (fd_forward_data[k] - residual_data[k]) * diff_step_inv;
              }
              with_respect_to = orig_wrt;
            }

            // save to last_res

            // push right away
            std::lock_guard<std::mutex> lock(residual_mutex);
            const auto& vdofs = *bed.v_dofs_;
            residual.AddElementVector(vdofs, tmp.local_residual_.GetData());
            double* A = grad.GetData();
            const double* local_A = tmp.local_grad_.GetData();

            const auto& A_ids = *bed.sparse_matrix_A_ids_;
            for (int k{}; k < A_ids.size(); ++k) {
              A[A_ids[k]] += *local_A++ * grad_factor;
            }
            if (save_residual) {
              // since we don't save our local copy, we can't really have this
              // serial at the end
              double* last_r_d = last_residual_.GetData();
              const double* local_r_d = tmp.local_residual_.GetData();
              for (int k{}; k < bed.n_tdof_; ++k) {
                last_r_d[bed.local_v_dofs_[k]] = local_r_d[k];
              }
            }
          }
          // reuse mutex for area and force update
          std::lock_guard<std::mutex> lock(residual_mutex);
          last_force_.Add(tl_force);
        };

    mimi::utils::NThreadExe(assemble_boundary_residual_and_grad_then_contribute,
                            n_marked_boundaries_,
                            (nthreads < 0) ? n_threads_ : nthreads);
  }

  /// checks GapNorm from given test_x
  virtual double GapNorm(const mfem::Vector& test_x, const int nthreads) {
    MIMI_FUNC()

    std::mutex gap_norm;
    double gap_squared_total{};
    auto find_gap = [&](const int begin, const int end, const int i_thread) {
      auto& tmp = temporary_data_[i_thread];
      double local_g_squared_sum{};
      // this loops marked boundary elements
      for (int i{begin}; i < end; ++i) {
        // get bed
        const BoundaryElementData& bed = boundary_element_data_[i];
        tmp.SetDof(bed.n_dof_);

        mfem::DenseMatrix& current_element_x =
            tmp.CurrentElementSolutionCopy(test_x, bed);

        for (const QuadData& q : bed.quad_data_) {
          // get current position and F
          current_element_x.MultTranspose(q.N_,
                                          tmp.distance_query_.query_.data());
          // query
          nearest_distance_coeff_->NearestDistance(tmp.distance_query_,
                                                   tmp.distance_results_);
          tmp.distance_results_.ComputeNormal<true>(); // unit normal
          const double g = tmp.distance_results_.NormalGap();
          if (g < 0.0) {
            local_g_squared_sum += g * g;
          }
        }
      } // marked elem loop
      {
        std::lock_guard<std::mutex> lock(gap_norm);
        gap_squared_total += local_g_squared_sum;
      }
    };

    mimi::utils::NThreadExe(find_gap,
                            n_marked_boundaries_,
                            (nthreads < 0) ? n_threads_ : nthreads);

    return std::sqrt(gap_squared_total);
  }

  virtual void BoundaryPostTimeAdvance(const mfem::Vector& current_x) {
    MIMI_FUNC()

    auto& rc = *RuntimeCommunication();
    if (rc.ShouldSave("contact_history")) {
      rc.RecordRealHistory("area", last_area_);
      rc.RecordRealHistory("force_x", last_force_[0]);
      rc.RecordRealHistory("force_y", last_force_[1]);
      if (dim_ > 2) {
        rc.RecordRealHistory("force_z", last_force_[2]);
      }
    }
    if (rc.ShouldSave("contact_forces")) {
      rc.SaveDynamicVector("vector_residual_", last_residual_);
    }
    if (rc.ShouldSave("")) {
    }
  }

  virtual void CreateMassMatrix() {
    MIMI_FUNC()

    // const int height = marked_boundary_v_dofs_.size() / dim_;
    const int height = marked_boundary_v_dofs_.size();
    // const int height = precomputed_->sparsity_pattern_->Height();

    if (m_mat_) {
      if (m_mat_->Height() != height) {
        m_mat_ = nullptr;
      }
    }

    if (!m_mat_) {
      // this is where we set local_dofs_
      m_mat_ = std::make_unique<mfem::SparseMatrix>(height);
      mfem::DenseMatrix v_elmat, elmat;
      for (auto& be : boundary_element_data_) {
        v_elmat.SetSize(be.n_tdof_, be.n_tdof_);
        v_elmat = 0.0;
        elmat.SetSize(be.n_dof_, be.n_dof_);
        elmat = 0.0;
        for (const auto& q_data : be.quad_data_) {
          mfem::AddMult_a_VVt(q_data.integration_weight_ * q_data.det_dX_dxi_,
                              q_data.N_,
                              elmat);
        }
        // be.local_dofs_.SetSize(be.n_dof_);
        // auto& vdof = *be.v_dofs_;
        // for (int i{}; i < be.n_dof_; ++i) {
        //   be.local_dofs_[i] = local_marked_dofs_[vdof[i] / dim_];
        // }
        elmat.Print();
        for (int d{}; d < dim_; ++d) {
          v_elmat.AddMatrix(elmat, be.n_dof_ * d, be.n_dof_ * d);
        }
        be.local_v_dofs_.Print();
        m_mat_->AddSubMatrix(be.local_v_dofs_, be.local_v_dofs_, v_elmat, 0);
      }
      m_mat_->Finalize();
      m_mat_->SortColumnIndices();
    }

    mass_inv_direct_.SetOperator(*m_mat_);
    mass_inv_direct_.SetPrintLevel(1);
  }

  virtual void L2Project(const mfem::Vector& vec, mfem::Vector& projected) {
    MIMI_FUNC()
    mass_inv_direct_.Mult(vec, projected);
  }

  virtual double LastGapNorm() const {
    MIMI_FUNC()

    double negative_gap_sum{};
    for (auto& be : boundary_element_data_) {
      for (auto& qd : be.quad_data_) {
        if (qd.g_ < 0.0) {
          negative_gap_sum += qd.g_ * qd.g_;
        }
      }
    }

    return std::sqrt(std::abs(negative_gap_sum));
  }
};

} // namespace mimi::integrators
