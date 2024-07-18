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
    bool active{false}; // needed for

    double integration_weight;
    double det_dX_dxi;

    double det_J;
    double lagrange;     // lambda_k
    double new_lagrange; // lambda_k+1 - this is also last_p
    double g;            // normal gap
    double old_g;        // normal gap

    inline void Record(const bool really,
                       const double detj,
                       const double new_lag,
                       const double g_in,
                       const bool active) {
      MIMI_FUNC()

      if (!really)
        return;
      det_J = detj;
      new_lagrange = new_lag;
      old_g = g;
      g = g_in;
      active = active;
    }

    mfem::Vector N; // shape
    /// thanks to Jac's hint, it turns out we can just work with this for
    /// boundary
    mfem::DenseMatrix dN_dxi; // shape grad
  };

  /// everything here is based on boundary element.
  /// consider that each variable has "boundary_" prefix
  struct BoundaryElementData {
    /// we only work with marked boundaries, so we keep two ids
    /// for convenience. One that belongs to marked boundary list (id)
    /// and one that belongs to NBE(true_id)
    int id;
    int true_id;
    int quadrature_order;
    int geometry_type;
    int n_quad;
    int n_dof; // this is not true dof
    int n_tdof;

    // we keep activity info in element level.
    bool active = false;

    std::shared_ptr<mfem::Array<int>> v_dofs;
    mfem::Array<int> local_v_dofs; // scalar dofs for L2Projection
    mfem::Array<int> local_dofs;   // scalar dofs for L2Projection

    Vector_<QuadData> quad_data;

    /// pointer to element and eltrans. don't need it,
    /// but maybe for further processing or something
    std::shared_ptr<mfem::NURBSFiniteElement> element;
    std::shared_ptr<mfem::FaceElementTransformations> element_trans;

    // direct access id to sparse matrix
    // as we only loop over marked boundaries, keep this pointer for easy access
    std::shared_ptr<mimi::utils::Vector<int>> sparse_matrix_A_ids;

    const mfem::IntegrationRule&
    GetIntRule(mfem::IntegrationRules& thread_int_rules) const {
      MIMI_FUNC()

      return thread_int_rules.Get(geometry_type, quadrature_order);
    }
  };
  /// temporary containers required in element assembly
  /// mfem performs some fancy checks for allocating memories.
  /// So we create one for each thread
  struct TemporaryData {
    /// wraps element_state_data_
    mfem::Vector element_x;
    /// wraps element_x
    mfem::DenseMatrix element_x_mat;
    /// wraps J_data
    mfem::DenseMatrix J;
    /// wraps forward_residual data
    mfem::DenseMatrix forward_residual;

    mimi::coefficients::NearestDistanceBase::Query distance_query;
    mimi::coefficients::NearestDistanceBase::Results distance_results;

    int dim{-1};

    void SetDim(const int dim_in) {
      MIMI_FUNC()

      dim = dim_in;
      assert(dim > 0);

      distance_query.SetSize(dim);
      distance_results.SetSize(dim - 1, dim);
    }

    void SetDof(const int n_dof) {
      MIMI_FUNC()

      assert(dim > 0);
      element_x.SetSize(n_dof * dim);
      element_x_mat.UseExternalData(element_x.GetData(), n_dof, dim);
      J.SetSize(dim, dim - 1);
      forward_residual.SetSize(n_dof, dim);
    }

    mfem::DenseMatrix&
    CurrentElementSolutionCopy(const mfem::Vector& current_all,
                               const BoundaryElementData& elem_data) {
      MIMI_FUNC()

      current_all.GetSubVector(*elem_data.v_dofs, element_x);
      return element_x_mat;
    }
  };

protected:
  /// scene
  std::shared_ptr<mimi::coefficients::NearestDistanceBase>
      nearest_distance_coeff_ = nullptr;

  /// convenient constants - space dim (dim) is in base
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
    boundary_geometry_type =
        precomputed_->boundary_elements_[0]->GetGeomType();

    boundary_element_data_.resize(n_marked_boundaries_);

    last_force_.Reallocate(dim_);

    const int default_q_order =
        RuntimeCommunication()->GetInteger("contact_quadrature_order", -1);

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
        i_bed.id = i;
        i_bed.true_id = m;

        // save (shared) pointers from global numbering
        i_bed.element = precomputed_->boundary_elements_[m];
        i_bed.geometry_type = i_bed.element->GetGeomType();
        i_bed.element_trans =
            precomputed_->reference_to_target_boundary_trans_[m];
        i_bed.n_dof = i_bed.element->GetDof();
        i_bed.n_tdof = i_bed.n_dof * dim_;
        i_bed.v_dofs = precomputed_->boundary_v_dofs_[m];
        // i_bed.dofs.SetSize(i_bed.v_dofs->Size() / dim_);
        // for (int k{}; l{}; k < i_bed.v_dofs->Size(); k += dim_, ++l) {
        //   i_bed.dofs[l] = (*i_bed.v_dofs)[k] / dim_;
        // }

        // quick size check
        const int n_tdof = i_bed.n_tdof;

        i_bed.sparse_matrix_A_ids = precomputed_->boundary_A_ids_[m];

        // let's prepare quad loop
        i_bed.quadrature_order = (default_q_order < 0)
                                      ? i_bed.element->GetOrder() * 2 + 3
                                      : default_q_order;

        const mfem::IntegrationRule& ir = i_bed.GetIntRule(int_rules);
        i_bed.n_quad = ir.GetNPoints();
        i_bed.quad_data.resize(i_bed.n_quad);

        // this needs to come from volume element!
        for (int j{}; j < i_bed.n_quad; ++j) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(j);
          i_bed.element_trans->SetIntPoint(&ip);

          // first allocate, then start filling values
          auto& q_data = i_bed.quad_data[j];
          q_data.integration_weight = ip.weight;
          q_data.N.SetSize(i_bed.n_dof);
          q_data.dN_dxi.SetSize(i_bed.n_dof, boundary_para_dim_);

          // precompute
          // shape comes from boundary element
          i_bed.element_->CalcShape(ip, q_data.N);
          i_bed.element_->CalcDShape(ip, q_data.dN_dxi);
          q_data.det_dX_dxi = i_bed.element_trans->Weight();

          // let's not forget to initialize values
          q_data.active = false;
          q_data.lagrange = 0.0;
          q_data.new_lagrange = 0.0;
          q_data.g = 0.0;
          q_data.old_g = 0.0;
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
      for (const int& vdof : *bed.v_dofs) {
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

    for (auto& bed : boundary_element_data_) {
      bed.local_v_dofs_.SetSize(bed.n_tdof);
      for (int i{}; i < bed.n_tdof; ++i) {
        bed.local_v_dofs_[i] = local_marked_v_dofs_[(*bed.v_dofs)[i]];
      }
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
  }

  // we will average lagrange
  virtual void UpdateLagrange() {
    MIMI_FUNC();

    for (auto& be : boundary_element_data_) {
      for (auto& qd : be.quad_data) {
        qd.lagrange = qd.new_lagrange;
      }
    }
  }

  virtual void FillLagrange(const double value) {
    MIMI_FUNC()

    for (auto& be : boundary_element_data_) {
      for (auto& qd : be.quad_data) {
        qd.lagrange = value;
        qd.new_lagrange = value;
      }
    }
  }

  bool
  QuadLoop(const mfem::DenseMatrix& x,
           Vector_<QuadData>& q_data,
           TemporaryData& tmp,
           mfem::DenseMatrix& residual_matrix,
           const bool keep_record,
           double& area, // must if keep_record
           mimi::utils::Data<double>& force /* only if applicable*/) const {
    MIMI_FUNC()
    bool any_active = false;
    const double penalty = nearest_distance_coeff_->coefficient_;

    double* residual_begin = residual_matrix.GetData();
    for (QuadData& q : q_data) {
      // get current position and F
      x.MultTranspose(q.N, tmp.distance_query.query_.data());

      // nearest distance query
      nearest_distance_coeff_->NearestDistance(tmp.distance_query,
                                               tmp.distance_results);
      tmp.distance_results.ComputeNormal<true>(); // unit normal
      const double g = tmp.distance_results.NormalGap();
      // for some reason, some query hit right at the very very end
      // and returned super big number / small number
      // Practical solution was to plant the tree with an odd number.
      assert(std::isfinite(g));

      // defer J eval if this isn't keep_record
      double det_J{};
      if (keep_record) {
        mfem::MultAtB(x, q.dN_dxi, tmp.J);
        det_J = tmp.J.Weight();
        area += det_J * q.integration_weight;
      }

      // check for angle and sign of normal gap for the first augmentation loop
      // otherwise, we detemine exit criteria based on p value.
      if (!(q.lagrange < 0.0)) {
        // normalgap validity and angle tolerance
        // angle between two vectors:  acos(a * b / (a.norm * b.norm))
        // difference * unit_normal is g, distance is denom.
        constexpr const double angle_tol = 1.e-5;
        if (g > 0.
            || std::acos(
                   std::min(1., std::abs(g) / tmp.distance_results.distance_))
                   > angle_tol) {
          q.Record(keep_record, det_J, 0.0, g, false);
          continue;
        }
      }

      // we allow reduction of p, as long as it doesn't change sign
      // implementation here has flipped sign so we cut at 0.0
      // see https://doi.org/10.1016/0045-7949(92)90540-G
      const double p = std::min(q.lagrange + penalty * g, 0.0);
      if (p == 0.0) {
        // this quad no longer contributes to contact enforcement
        q.Record(keep_record, det_J, 0.0, g, false);
        continue;
      }

      // do deferred
      if (!keep_record) {
        mfem::MultAtB(x, q.dN_dxi, tmp.J);
        det_J = tmp.J.Weight();
      }

      // maybe only a minor difference, but we don't want to keep records from
      // line search or FD computations
      q.Record(keep_record, det_J, p, g, true);

      // set active after checking p
      any_active = true;

      // again, note no negative sign.
      AddMult_a_VWt(q.integration_weight * det_J * p,
                    q.N.begin(),
                    q.N.end(),
                    tmp.distance_results.normal_.begin(),
                    tmp.distance_results.normal_.end(),
                    residual_begin);
      if (keep_record) {
        force.Add(p * det_J * q.integration_weight,
                  tmp.distance_results.normal_.begin());
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

    std::mutex residual_mutex;

    auto assemble_face_residual_and_maybe_grad =
        [&](const int begin, const int end, const int i_thread) {
          TemporaryData tmp;
          tmp.SetDim(dim_);
          double area_dummy;
          mimi::utils::Data<double> force_dummy;
          // this loops marked boundary elements
          for (int i{begin}; i < end; ++i) {
            // get bed
            BoundaryElementData& bed = boundary_element_data_[i];
            tmp.SetDof(bed.n_dof);
            // variable name is misleading - this is just local residual
            // we use this container, as we already allocate this in tmp
            // anyways
            tmp.forward_residual = 0.0;

            mfem::DenseMatrix& current_element_x =
                tmp.CurrentElementSolutionCopy(current_x, bed);

            // assemble residual
            // this function is called either at the last step at "melted"
            // staged or during line search, keep_record = False
            const bool any_active = QuadLoop(current_element_x,
                                             bed.quad_data,
                                             tmp,
                                             tmp.forward_residual,
                                             false,
                                             area_dummy,
                                             force_dummy);

            // only push if any of quad point was active
            if (any_active) {
              const std::lock_guard<std::mutex> lock(residual_mutex);
              residual.AddElementVector(*bed.v_dofs,
                                        tmp.forward_residual.GetData());
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
    std::mutex residual_mutex;
    // lambda for nthread assemble
    auto assemble_boundary_residual_and_grad_then_contribute =
        [&](const int begin, const int end, const int i_thread) {
          mfem::DenseMatrix local_residual;
          mfem::DenseMatrix local_grad;
          TemporaryData tmp;
          tmp.SetDim(dim_);
          double area_dummy;
          mimi::utils::Data<double> force_dummy;
          for (int i{begin}; i < end; ++i) {
            BoundaryElementData& bed = boundary_element_data_[i];
            tmp.SetDof(bed.n_dof);
            local_residual.SetSize(bed.n_dof, dim_);
            local_grad.SetSize(bed.n_tdof, bed.n_tdof);
            local_residual = 0.0;

            // get current element solution as matrix
            mfem::DenseMatrix& current_element_x =
                tmp.CurrentElementSolutionCopy(current_x, bed);

            // assemble residual
            const bool any_active = QuadLoop(current_element_x,
                                             bed.quad_data,
                                             tmp,
                                             local_residual,
                                             false,
                                             area_dummy,
                                             force_dummy);

            // skip if nothing's active
            if (!any_active) {
              continue;
            }

            double* grad_data = local_grad.GetData();
            double* solution_data = current_element_x.GetData();
            const double* residual_data = local_residual.GetData();
            const double* fd_forward_data = tmp.forward_residual.GetData();
            for (int j{}; j < bed.n_tdof; ++j) {
              tmp.forward_residual = 0.0;

              double& with_respect_to = *solution_data++;
              const double orig_wrt = with_respect_to;
              const double diff_step = std::abs(orig_wrt) * 1.0e-8;
              const double diff_step_inv = 1. / diff_step;

              with_respect_to = orig_wrt + diff_step;
              QuadLoop(current_element_x,
                       bed.quad_data,
                       tmp,
                       tmp.forward_residual,
                       false,
                       area_dummy,
                       force_dummy);

              for (int k{}; k < bed.n_tdof; ++k) {
                *grad_data++ =
                    (fd_forward_data[k] - residual_data[k]) * diff_step_inv;
              }
              with_respect_to = orig_wrt;
            }

            // push right away
            std::lock_guard<std::mutex> lock(residual_mutex);
            double* A = grad.GetData();
            const double* local_A = local_grad.GetData();
            const auto& A_ids = *bed.sparse_matrix_A_ids;
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
          mfem::DenseMatrix local_residual;
          mfem::DenseMatrix local_grad;
          TemporaryData tmp;
          tmp.SetDim(dim_);
          double tl_area{};
          mimi::utils::Data<double> tl_force(dim_);
          for (int i{begin}; i < end; ++i) {
            BoundaryElementData& bed = boundary_element_data_[i];
            tmp.SetDof(bed.n_dof);
            local_residual.SetSize(bed.n_dof, dim_);
            local_grad.SetSize(bed.n_tdof, bed.n_tdof);
            local_residual = 0.0;

            // get current element solution as matrix
            mfem::DenseMatrix& current_element_x =
                tmp.CurrentElementSolutionCopy(current_x, bed);

            // assemble residual
            const bool any_active = QuadLoop(current_element_x,
                                             bed.quad_data,
                                             tmp,
                                             local_residual,
                                             true,
                                             tl_area,
                                             tl_force);

            // skip if nothing's active
            if (!any_active) {
              continue;
            }

            double* grad_data = local_grad.GetData();
            double* solution_data = current_element_x.GetData();
            const double* residual_data = local_residual.GetData();
            const double* fd_forward_data = tmp.forward_residual.GetData();
            for (int j{}; j < bed.n_tdof; ++j) {
              tmp.forward_residual = 0.0;

              double& with_respect_to = *solution_data++;
              const double orig_wrt = with_respect_to;
              const double diff_step = std::abs(orig_wrt) * 1.0e-8;
              const double diff_step_inv = 1. / diff_step;

              with_respect_to = orig_wrt + diff_step;
              QuadLoop(current_element_x,
                       bed.quad_data,
                       tmp,
                       tmp.forward_residual,
                       false,
                       tl_area, /* this won't be changed */
                       tl_force /* this either */);

              for (int k{}; k < bed.n_tdof; ++k) {
                *grad_data++ =
                    (fd_forward_data[k] - residual_data[k]) * diff_step_inv;
              }
              with_respect_to = orig_wrt;
            }

            // save to last_res

            // push right away
            std::lock_guard<std::mutex> lock(residual_mutex);
            const auto& vdofs = *bed.v_dofs;
            residual.AddElementVector(vdofs, local_residual.GetData());
            double* A = grad.GetData();
            const double* local_A = local_grad.GetData();
            const auto& A_ids = *bed.sparse_matrix_A_ids;
            for (int k{}; k < A_ids.size(); ++k) {
              A[A_ids[k]] += *local_A++ * grad_factor;
            }
            if (save_residual) {
              // since we don't save our local copy, we can't really have this
              // serial at the end
              double* last_r_d = last_residual_.GetData();
              const double* local_r_d = local_residual.GetData();
              for (int k{}; k < bed.n_tdof; ++k) {
                last_r_d[bed.local_v_dofs_[k]] = local_r_d[k];
              }
            }
          }
          // reuse mutex for area and force update
          std::lock_guard<std::mutex> lock(residual_mutex);
          last_area_ += tl_area;
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
      TemporaryData tmp;
      tmp.SetDim(dim_);
      double local_g_squared_sum{};
      // this loops marked boundary elements
      for (int i{begin}; i < end; ++i) {
        // get bed
        const BoundaryElementData& bed = boundary_element_data_[i];
        tmp.SetDof(bed.n_dof);

        mfem::DenseMatrix& current_element_x =
            tmp.CurrentElementSolutionCopy(test_x, bed);

        for (const QuadData& q : bed.quad_data) {
          // get current position and F
          current_element_x.MultTranspose(q.N,
                                          tmp.distance_query.query_.data());
          // query
          nearest_distance_coeff_->NearestDistance(tmp.distance_query,
                                                   tmp.distance_results);
          tmp.distance_results.ComputeNormal<true>(); // unit normal
          const double g = tmp.distance_results.NormalGap();
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
        v_elmat.SetSize(be.n_tdof, be.n_tdof);
        v_elmat = 0.0;
        elmat.SetSize(be.n_dof, be.n_dof);
        elmat = 0.0;
        for (const auto& q_data : be.quad_data) {
          mfem::AddMult_a_VVt(q_data.integration_weight * q_data.det_dX_dxi,
                              q_data.N,
                              elmat);
        }
        // be.local_dofs_.SetSize(be.n_dof);
        // auto& vdof = *be.v_dofs;
        // for (int i{}; i < be.n_dof; ++i) {
        //   be.local_dofs_[i] = local_marked_dofs_[vdof[i] / dim_];
        // }
        elmat.Print();
        for (int d{}; d < dim_; ++d) {
          v_elmat.AddMatrix(elmat, be.n_dof * d, be.n_dof * d);
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
      for (auto& qd : be.quad_data) {
        if (qd.g < 0.0) {
          negative_gap_sum += qd.g * qd.g;
        }
      }
    }

    return std::sqrt(std::abs(negative_gap_sum));
  }
};

} // namespace mimi::integrators
