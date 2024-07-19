#pragma once

#include <cmath>
#include <fstream>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/coefficients/nearest_distance.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/precomputed.hpp"
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
/// A large deformation frictional contact formulation using NURBS-based
/// isogeometric analysis"
/// Currently implements normal contact
class MortarContact : public NonlinearBase {
protected:
  static const int kDof{0};
  static const int kVDof{1};

  static const int kXRef{0};

public:
  using Base_ = NonlinearBase;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;
  using ElementQuadData_ = mimi::utils::ElementQuadData;
  using ElementData_ = mimi::utils::ElementData;
  using QuadData_ = mimi::utils::QuadData;

  // post process
  std::unique_ptr<mfem::SparseMatrix> m_mat_;
  mfem::CGSolver mass_inv_;
  mfem::DSmoother mass_inv_prec_;
  mfem::UMFPackSolver mass_inv_direct_;
  mfem::Vector mass_b_; // for rhs
  mfem::Vector mass_x_; // for answer
  double last_area_;    // for p = forces / A
  double last_pressure_;
  mimi::utils::Data<double> last_force_;

  /// temporary containers required in element assembly
  /// mfem performs some fancy checks for allocating memories.
  /// So we create one for each thread
  class TemporaryData {
  public:
    mfem::Vector element_x_;
    mfem::DenseMatrix element_x_mat_;
    mfem::DenseMatrix J_;
    mfem::DenseMatrix local_residual_;
    mfem::DenseMatrix local_grad_;
    mfem::DenseMatrix forward_residual_;

    mfem::Vector element_pressure_;

    mimi::coefficients::NearestDistanceBase::Query distance_query_;
    mimi::coefficients::NearestDistanceBase::Results distance_results_;

    mfem::Vector normal_;

    // thread local
    mfem::Vector local_area_;
    mfem::Vector local_gap_;
    // global, read as thread_local (pause) area
    mfem::Vector thread_local_area_;
    mfem::Vector thread_local_gap_;

    int dim_{-1};
    int n_dof_{-1};
    bool assembling_grad_{false};

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
      n_dof_ = n_dof;
      element_x_.SetSize(n_dof * dim_);
      element_x_mat_.UseExternalData(element_x_.GetData(), n_dof, dim_);
      local_residual_.SetSize(n_dof, dim_);
      local_grad_.SetSize(n_dof * dim_, n_dof * dim_);
      forward_residual_.SetSize(n_dof, dim_);
      element_pressure_.SetSize(n_dof);
      local_area_.SetSize(n_dof);
      local_gap_.SetSize(n_dof);
    }

    mfem::DenseMatrix&
    CurrentElementSolutionCopy(const mfem::Vector& current_all,
                               const ElementQuadData_& eq_data) {
      MIMI_FUNC()

      current_all.GetSubVector(eq_data.GetElementData().v_dofs, element_x_);
      element_x_mat_ += eq_data.GetMatrix(kXRef);

      return element_x_mat_;
    }

    mfem::Vector& CurrentElementPressure(const mfem::Vector& pressure_all,
                                         const ElementQuadData_& eq_data) {
      MIMI_FUNC()

      pressure_all.GetSubVector(eq_data.GetArray(kDof), element_pressure_);
      return element_pressure_;
    }

    mfem::DenseMatrix& ComputeJ(const mfem::DenseMatrix& dndxi) {
      MIMI_FUNC()
      mfem::MultAtB(element_x_mat_, dndxi, J_);
      return J_;
    }

    void ComputeNearestDistanceQuery(const mfem::Vector& N) {
      MIMI_FUNC()
      element_x_mat_.MultTranspose(N, distance_query_.query.data());
    }

    bool IsPressureZero() const {
      for (const double p : element_pressure_) {
        if (p != 0.0) {
          return false;
        }
      }
      return true;
    }

    double Pressure(const mfem::Vector& N) const {
      MIMI_FUNC()
      const double* p_d = element_pressure_.GetData();
      const double* n_d = N.GetData();
      double p{};
      for (int i{}; i < N.Size(); ++i) {
        p += n_d[i] * p_d[i];
      }
      return p;
    }

    void InitializeThreadLocalAreaAndGap() {
      double* a_d = thread_local_area_.GetData();
      double* g_d = thread_local_gap_.GetData();
      for (int i{}; i < thread_local_area_.Size(); ++i) {
        a_d[i] = 0.0;
        g_d[i] = 0.0;
      }
    }

    void GradAssembly(bool state) {
      MIMI_FUNC()

      assembling_grad_ = state;
    }

    mfem::DenseMatrix& ResidualMatrix() {
      MIMI_FUNC()
      if (assembling_grad_) {
        return forward_residual_;
      }
      return local_residual_;
    }

    int GetTDof() const { return dim_ * n_dof_; }

    mfem::DenseMatrix& CurrentSolution() {
      MIMI_FUNC()

      return element_x_mat_;
    }
  };

protected:
  /// scene
  std::shared_ptr<mimi::coefficients::NearestDistanceBase>
      nearest_distance_coeff_ = nullptr;

  /// convenient constants - space dim (dim_) is in base
  int dim_;
  int boundary_para_dim_;
  int n_threads_;
  int n_marked_boundaries_;

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
  mfem::Vector average_pressure_;
  mfem::Vector area_;

  Vector_<TemporaryData> temporary_data_;

public:
  MortarContact(
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

    // get sizes
    dim_ = precomputed_->dim_;
    boundary_para_dim_ = dim_ - 1;
    n_threads_ = precomputed_->n_threads_;
    // this is actual number of boundaries that contributes to assembly
    n_marked_boundaries_ = Base_::marked_boundary_elements_.size();

    last_force_.Reallocate(dim_);
    temporary_data_.resize(n_threads_);

    Vector_<ElementQuadData_>& element_quad_data_vec =
        precomputed_->GetElementQuadData(Name());

    // reserve enough space - supports can overlap so, be generous initially
    marked_boundary_v_dofs_.clear();
    marked_boundary_v_dofs_.reserve(
        n_marked_boundaries_ * dim_
        * std::pow((precomputed_->fe_spaces_[0]->GetMaxElementOrder() + 1),
                   boundary_para_dim_));
    for (const ElementQuadData_& eqd : element_quad_data_vec) {
      for (const int& vdof : eqd.GetElementData().v_dofs) {
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
    average_pressure_.SetSize(local_dof_size);
    average_pressure_ = 0.0;
    area_.SetSize(local_dof_size);
    area_ = 0.0;

    auto dofs_and_x_ref_thread_local =
        [&](const int begin, const int end, const int i_thread) {
          // prepare local dofs for local, reduced length vectors
          for (int i{begin}; i < end; ++i) {
            ElementQuadData_& eqd = element_quad_data_vec[i];
            ElementData_& ed = eqd.GetElementData();

            // get x_ref -> transposed layout
            eqd.MakeMatrices(1);
            mfem::DenseMatrix& x_ref = eqd.GetMatrix(kXRef);
            x_ref.SetSize(ed.n_dof, dim_);
            const mfem::DenseMatrix& p_mat = ed.element_trans->GetPointMat();
            for (int i_dof{}; i_dof < ed.n_dof; ++i_dof) {
              for (int i_dim{}; i_dim < dim_; ++i_dim) {
                x_ref(i_dof, i_dim) = p_mat(i_dim, i_dof);
              }
            }

            // get local dofs
            // v_dofs are xxxxyyyyzzzz
            eqd.MakeArrays(2);
            mfem::Array<int>& local_dofs = eqd.GetArray(kDof);
            mfem::Array<int>& local_vdofs = eqd.GetArray(kVDof);
            local_dofs.SetSize(ed.n_dof);
            local_vdofs.SetSize(ed.n_tdof);
            for (int i{}; i < ed.n_tdof; ++i) {
              if (i < ed.n_dof) {
                local_dofs[i] = local_marked_dofs_[ed.v_dofs[i] / dim_];
              }
              local_vdofs[i] = local_marked_v_dofs_[ed.v_dofs[i]];
            }
          }
          TemporaryData& tmp = temporary_data_[i_thread];
          tmp.SetDim(precomputed_->dim_);
          tmp.thread_local_area_.SetSize(local_dof_size);
          tmp.thread_local_gap_.SetSize(local_dof_size);
        };

    mimi::utils::NThreadExe(dofs_and_x_ref_thread_local,
                            n_marked_boundaries_,
                            n_threads_);
  }

  void InitializeGapAreaPressure() {
    MIMI_FUNC()

    double* ag_d = average_gap_.GetData();
    double* ap_d = average_pressure_.GetData();
    double* ad = area_.GetData();
    for (int i{}; i < average_gap_.Size(); ++i) {
      ad[i] = 0.0;
      ag_d[i] = 0.0;
      ap_d[i] = 0.0;
    }
  }

  void ElementGapAndArea(const Vector_<QuadData_>& q_data,
                         TemporaryData& tmp,
                         double& area_contribution) const {

    tmp.local_area_ = 0.0;
    tmp.local_gap_ = 0.0;
    double* la_d = tmp.local_area_.GetData();
    double* lg_d = tmp.local_gap_.GetData();
    const int N_size = q_data[0].N.Size();

    for (const QuadData_& q : q_data) {
      // compute query, then distance, then normal
      tmp.ComputeNearestDistanceQuery(q.N);
      nearest_distance_coeff_->NearestDistance(tmp.distance_query_,
                                               tmp.distance_results_);
      tmp.distance_results_.ComputeNormal<true>(); // unit normal

      // get normal gaps and make sure it's not negative
      const double true_g = tmp.distance_results_.NormalGap();
      double g = std::min(true_g, 0.);

      // jacobian
      mfem::DenseMatrix& J = tmp.ComputeJ(q.dN_dxi);
      const double fac = q.integration_weight * J.Weight();
      area_contribution += fac;

      // normalgap validity and angle tolerance
      // angle between two vectors:  acos(a * b / (a.norm * b.norm))
      constexpr const double angle_tol = 1.e-5;
      if (std::acos(
              std::min(1., std::abs(true_g) / tmp.distance_results_.distance_))
          > angle_tol) {
        g = 0.0;
      }

      const double* N_d = q.N.GetData();
      const double fac_g = fac * g; /* maybe zero */
      for (int i{}; i < N_size; ++i) {
        const double N = N_d[i];
        la_d[i] += fac * N;
        lg_d[i] += fac_g * N;
      }
    }
  }

  /// Computes average gap. This loops over the whole boundary as in our
  /// application, we need to compute current area and it's done here.
  void ComputePressure(const mfem::Vector& current_u, double& area) {
    MIMI_FUNC()

    // first assemble thread locally
    std::mutex gap_mutex;
    auto prepare_average_gap =
        [&](const int begin, const int end, const int i_thread) {
          // deref
          Vector_<ElementQuadData_>& element_quad_data_vec =
              precomputed_->GetElementQuadData(Name());
          double local_area{};
          auto& tmp = temporary_data_[i_thread];
          // initialize thread local vectors
          tmp.InitializeThreadLocalAreaAndGap();
          for (int i{begin}; i < end; ++i) {
            // deref and prepare
            ElementQuadData_& eqd = element_quad_data_vec[i];
            ElementData_& bed = eqd.GetElementData();
            tmp.SetDof(bed.n_dof);
            tmp.CurrentElementSolutionCopy(current_u, eqd);

            ElementGapAndArea(eqd.GetQuadData(), tmp, local_area);

            // thread local push
            tmp.thread_local_area_.AddElementVector(eqd.GetArray(kDof),
                                                    tmp.local_area_);
            tmp.thread_local_gap_.AddElementVector(eqd.GetArray(kDof),
                                                   tmp.local_gap_);
          }
          // area contribution
          std::lock_guard<std::mutex> lock(gap_mutex);
          area += local_area;
        };

    mimi::utils::NThreadExe(prepare_average_gap,
                            n_marked_boundaries_,
                            n_threads_);

    // compute pressure by reducing and dividing
    InitializeGapAreaPressure();
    auto compute_pressure = [&](const int begin,
                                const int end,
                                const int i_thread) {
      const double penalty = nearest_distance_coeff_->coefficient_;
      const int size = end - begin;
      // destination
      double* gap_begin = average_gap_.GetData() + begin;
      double* p_begin = average_pressure_.GetData() + begin;
      double* area_begin = area_.GetData() + begin;
      // first we reduce
      for (const TemporaryData& tmp : temporary_data_) {
        const double* tl_area_begin = tmp.thread_local_area_.GetData() + begin;
        const double* tl_gap_begin = tmp.thread_local_gap_.GetData() + begin;
        for (int i{}; i < size; ++i) {
          area_begin[i] += tl_area_begin[i];
          gap_begin[i] += tl_gap_begin[i];
        }
      }

      for (int i{}; i < size; ++i) {
        // at average gap, we've already filtered inactive gaps
        // so all we need to do is just to multiply
        p_begin[i] = gap_begin[i] / area_begin[i] * penalty;
      }
    };

    mimi::utils::NThreadExe(compute_pressure, area_.Size(), n_threads_);
  }

  /// assembles contact traction with pressure. make sure pressure is non zero
  /// (IsPressureZero), otherwise this will be a waste
  template<bool record>
  void ElementResidual(const Vector_<QuadData_>& q_data,
                       TemporaryData& tmp,
                       mimi::utils::Data<double>& force /* only if applicable*/,
                       double& pressure_integral) const {
    MIMI_FUNC()
    tmp.ResidualMatrix() = 0.0;
    const double penalty = nearest_distance_coeff_->coefficient_;
    double* residual_d = tmp.ResidualMatrix().GetData();

    for (const QuadData_& q : q_data) {
      const double p = tmp.Pressure(q.N);
      assert(std::isfinite(p));
      assert(p < 0.);

      // we need derivatives to get normal.
      mfem::DenseMatrix& J = tmp.ComputeJ(q.dN_dxi);
      const double det_J = J.Weight();
      ComputeUnitNormal(J, tmp.normal_);

      // this one has negative sign as we use the normal from gauss point
      const double fac = q.integration_weight * det_J * p;
      Ptr_AddMult_a_VWt(-fac,
                        q.N.begin(),
                        q.N.end(),
                        tmp.normal_.begin(),
                        tmp.normal_.end(),
                        residual_d);

      // this is application specific quantity
      if constexpr (record) {
        force.Add(fac, tmp.normal_.begin());
        pressure_integral += fac;
      }
    }
  }

  /// Same rule applies as ElementResidual - check if pressure is non zero.
  /// Note - we can probably generalize this with fold expressions for all
  /// integrators
  void ElementResidualAndGrad(
      const Vector_<QuadData_>& q_data,
      TemporaryData& tmp,
      mimi::utils::Data<double>& force /* only if applicable*/,
      double& pressure_integral) {
    MIMI_FUNC()

    tmp.GradAssembly(false);
    ElementResidual<true>(q_data, tmp, force, pressure_integral);

    tmp.GradAssembly(true);
    mimi::utils::Data<double> dummy_f;
    double dummy_p;
    double* grad_data = tmp.local_grad_.GetData();
    double* solution_data = tmp.CurrentSolution().GetData();
    const double* residual_data = tmp.local_residual_.GetData();
    const double* fd_forward_data = tmp.forward_residual_.GetData();
    const int n_t_dof = tmp.GetTDof();
    for (int i{}; i < n_t_dof; ++i) {
      double& with_respect_to = *solution_data++;
      const double orig_wrt = with_respect_to;
      const double diff_step =
          (orig_wrt != 0.0) ? std::abs(orig_wrt) * 1.0e-8 : 1.0e-10;
      const double diff_step_inv = 1. / diff_step;

      with_respect_to = orig_wrt + diff_step;
      ElementResidual<false>(q_data, tmp, dummy_f, dummy_p);
      for (int j{}; j < n_t_dof; ++j) {
        *grad_data++ = (fd_forward_data[j] - residual_data[j]) * diff_step_inv;
      }
      with_respect_to = orig_wrt;
    }
  }

  /// rhs (for us, we have them on lhs - thus fliped sign) - used in line
  /// search
  virtual void AddBoundaryResidual(const mfem::Vector& current_u,
                                   mfem::Vector& residual) {
    MIMI_FUNC()

    last_area_ = 0.0;
    ComputePressure(current_u, last_area_);
    last_force_.Fill(0.0);
    last_pressure_ = 0.0;

    std::mutex residual_mutex;
    auto assemble_face_residual = [&](const int begin,
                                      const int end,
                                      const int i_thread) {
      auto& tmp = temporary_data_[i_thread];
      tmp.GradAssembly(false);
      mimi::utils::Data<double> tl_force(dim_);
      tl_force.Fill(0.0);
      double local_pressure{};
      Vector_<ElementQuadData_>& element_quad_data_vec =
          precomputed_->GetElementQuadData(Name());

      // this loops marked boundary elements
      for (int i{begin}; i < end; ++i) {
        ElementQuadData_& beqd = element_quad_data_vec[i];

        // first we check if we need to do anything - if pressure is zero,
        // nothing to do
        tmp.CurrentElementPressure(average_pressure_, beqd);
        if (tmp.IsPressureZero()) {
          continue;
        }

        // continue with assembly
        ElementData_& bed = beqd.GetElementData();
        tmp.SetDof(bed.n_dof);

        tmp.CurrentElementSolutionCopy(current_u, beqd);
        ElementResidual<true>(beqd.GetQuadData(),
                              tmp,
                              tl_force,
                              local_pressure);

        {
          const std::lock_guard<std::mutex> lock(residual_mutex);
          residual.AddElementVector(bed.v_dofs, tmp.local_residual_.GetData());
        }
      } // marked elem loop
        // reuse mutex for area and force update
      std::lock_guard<std::mutex> lock(residual_mutex);
      last_force_.Add(tl_force);
      last_pressure_ += local_pressure;
    };

    mimi::utils::NThreadExe(assemble_face_residual,
                            n_marked_boundaries_,
                            n_threads_);
  }

  virtual void AddBoundaryGrad(const mfem::Vector& current_u,
                               mfem::SparseMatrix& grad) const {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError(
        "Currently not implemented, use AddDomainResidualAndGrad");
  }

  /// assembles residual and grad at the same time. this should also have
  /// assembled the last converged residual
  virtual void AddBoundaryResidualAndGrad(const mfem::Vector& current_u,
                                          const double grad_factor,
                                          mfem::Vector& residual,
                                          mfem::SparseMatrix& grad) {
    MIMI_FUNC()

    last_area_ = 0.0;
    ComputePressure(current_u, last_area_);
    last_force_.Fill(0.0);
    last_pressure_ = 0.0;

    std::mutex residual_mutex;
    // lambda for nthread assemble
    auto assemble_boundary_residual_and_grad_then_contribute =
        [&](const int begin, const int end, const int i_thread) {
          auto& tmp = temporary_data_[i_thread];
          mimi::utils::Data<double> tl_force(dim_);
          tl_force.Fill(0.0);
          double local_pressure{};
          Vector_<ElementQuadData_>& element_quad_data_vec =
              precomputed_->GetElementQuadData(Name());
          for (int i{begin}; i < end; ++i) {
            ElementQuadData_& beqd = element_quad_data_vec[i];

            // first we check if we need to do anything - if pressure is zero,
            // nothing to do
            tmp.CurrentElementPressure(average_pressure_, beqd);
            if (tmp.IsPressureZero()) {
              continue;
            }

            ElementData_& bed = beqd.GetElementData();
            tmp.SetDof(bed.n_dof);

            // get current element solution as matrix
            tmp.CurrentElementSolutionCopy(current_u, beqd);
            ElementResidualAndGrad(beqd.GetQuadData(),
                                   tmp,
                                   tl_force,
                                   local_pressure);

            // push right away
            std::lock_guard<std::mutex> lock(residual_mutex);
            const auto& vdofs = bed.v_dofs;
            residual.AddElementVector(vdofs, tmp.local_residual_.GetData());
            double* A = grad.GetData();
            const double* local_A = tmp.local_grad_.GetData();
            const auto& A_ids = bed.A_ids;
            for (int k{}; k < A_ids.size(); ++k) {
              A[A_ids[k]] += *local_A++ * grad_factor;
            }
          }
          // reuse mutex for area and force update
          std::lock_guard<std::mutex> lock(residual_mutex);
          last_force_.Add(tl_force);
          last_pressure_ += local_pressure;
        };

    mimi::utils::NThreadExe(assemble_boundary_residual_and_grad_then_contribute,
                            n_marked_boundaries_,
                            n_threads_);
  }

  /// checks GapNorm from given test_x
  virtual double GapNorm(const mfem::Vector& test_x, const int nthreads) {
    MIMI_FUNC()

    std::mutex gap_norm;
    double gap_squared_total{};
    auto find_gap = [&](const int begin, const int end, const int i_thread) {
      auto& tmp = temporary_data_[i_thread];
      double local_g_squared_sum{};
      Vector_<ElementQuadData_>& element_quad_data_vec =
          precomputed_->GetElementQuadData(Name());
      // this loops marked boundary elements
      for (int i{begin}; i < end; ++i) {
        // get bed
        const ElementQuadData_& beqd = element_quad_data_vec[i];
        const ElementData_& bed = beqd.GetElementData();
        tmp.SetDof(bed.n_dof);

        mfem::DenseMatrix& current_element_x =
            tmp.CurrentElementSolutionCopy(test_x, beqd);

        for (const QuadData_& q : beqd.GetQuadData()) {
          // get current position and F
          current_element_x.MultTranspose(q.N,
                                          tmp.distance_query_.query.data());
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

  virtual void BoundaryPostTimeAdvance(const mfem::Vector& current_u) {
    MIMI_FUNC()
    auto& rc = *RuntimeCommunication();
    if (rc.ShouldSave("contact_history")) {
      rc.RecordRealHistory("area", last_area_);
      rc.RecordRealHistory("force_x", last_force_[0]);
      rc.RecordRealHistory("force_y", last_force_[1]);
      rc.RecordRealHistory("x_over_y", last_force_[0] / last_force_[1]);
      rc.RecordRealHistory("pressure", last_pressure_);
      if (dim_ > 2) {
        rc.RecordRealHistory("force_z", last_force_[2]);
      }
    }
    if (rc.ShouldSave("contact_forces")) {
      rc.SaveDynamicVector(
          "pressure_",
          average_pressure_); // with this we can always rebuild traction
    }
  }
};

} // namespace mimi::integrators
