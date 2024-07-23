#include "mimi/integrators/mortar_contact.hpp"

namespace mimi::integrators {

static const int kDof = MortarContactWorkData::kDof;
static const int kVDof = MortarContactWorkData::kVDof;
static const int kXRef = MortarContactWorkData::kXRef;

void MortarContact::Prepare() {
  MIMI_FUNC()

  // get sizes
  dim_ = precomputed_->dim_;
  boundary_para_dim_ = dim_ - 1;
  n_threads_ = precomputed_->n_threads_;
  // this is actual number of boundaries that contributes to assembly
  n_marked_boundaries_ = Base_::marked_boundary_elements_.size();

  last_force_.Reallocate(dim_);
  work_data_.resize(n_threads_);

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
          for (int j{}; j < ed.n_tdof; ++j) {
            if (j < ed.n_dof) {
              local_dofs[j] = local_marked_dofs_[ed.v_dofs[j] / dim_];
            }
            local_vdofs[j] = local_marked_v_dofs_[ed.v_dofs[j]];
          }
        }
        MortarContactWorkData& w = work_data_[i_thread];
        w.SetDim(precomputed_->dim_);
        w.thread_local_area_.SetSize(local_dof_size);
        w.thread_local_gap_.SetSize(local_dof_size);
      };

  mimi::utils::NThreadExe(dofs_and_x_ref_thread_local,
                          n_marked_boundaries_,
                          n_threads_);
}

void MortarContact::InitializeGapAreaPressure() {
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

void MortarContact::ElementGapAndArea(const Vector_<QuadData_>& q_data,
                                      MortarContactWorkData& w,
                                      double& area_contribution) const {

  w.local_area_ = 0.0;
  w.local_gap_ = 0.0;
  double* la_d = w.local_area_.GetData();
  double* lg_d = w.local_gap_.GetData();
  const int N_size = q_data[0].N.Size();

  for (const QuadData_& q : q_data) {
    // compute query, then distance, then normal
    w.ComputeNearestDistanceQuery(q.N);
    nearest_distance_coeff_->NearestDistance(w.distance_query_,
                                             w.distance_results_);
    w.distance_results_.ComputeNormal<true>(); // unit normal

    // get normal gaps and make sure it's not negative
    const double true_g = w.distance_results_.NormalGap();
    double g = std::min(true_g, 0.);

    // jacobian
    mfem::DenseMatrix& J = w.ComputeJ(q.dN_dxi);
    const double fac = q.integration_weight * J.Weight();
    area_contribution += fac;

    // normalgap validity and angle tolerance
    // angle between two vectors:  acos(a * b / (a.norm * b.norm))
    constexpr const double angle_tol = 1.e-5;
    if (std::acos(
            std::min(1., std::abs(true_g) / w.distance_results_.distance_))
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
void MortarContact::ComputePressure(const mfem::Vector& current_u,
                                    double& area) {
  MIMI_FUNC()

  // first assemble thread locally
  std::mutex gap_mutex;
  auto prepare_average_gap = [&](const int begin,
                                 const int end,
                                 const int i_thread) {
    // deref
    Vector_<ElementQuadData_>& element_quad_data_vec =
        precomputed_->GetElementQuadData(Name());
    double local_area{};
    auto& w = work_data_[i_thread];
    // initialize thread local vectors
    w.InitializeThreadLocalAreaAndGap();
    for (int i{begin}; i < end; ++i) {
      // deref and prepare
      ElementQuadData_& eqd = element_quad_data_vec[i];
      ElementData_& bed = eqd.GetElementData();
      w.SetDof(bed.n_dof);
      w.CurrentElementSolutionCopy(current_u, eqd);

      ElementGapAndArea(eqd.GetQuadData(), w, local_area);

      // thread local push
      w.thread_local_area_.AddElementVector(eqd.GetArray(kDof), w.local_area_);
      w.thread_local_gap_.AddElementVector(eqd.GetArray(kDof), w.local_gap_);
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
  auto compute_pressure =
      [&](const int begin, const int end, const int i_thread) {
        const double penalty = nearest_distance_coeff_->coefficient_;
        const int size = end - begin;
        // destination
        double* gap_begin = average_gap_.GetData() + begin;
        double* p_begin = average_pressure_.GetData() + begin;
        double* area_begin = area_.GetData() + begin;
        // first we reduce
        for (const MortarContactWorkData& w : work_data_) {
          const double* tl_area_begin = w.thread_local_area_.GetData() + begin;
          const double* tl_gap_begin = w.thread_local_gap_.GetData() + begin;
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

void MortarContact::ElementResidualAndGrad(
    const Vector_<QuadData_>& q_data,
    MortarContactWorkData& w,
    mimi::utils::Data<double>& force /* only if applicable*/,
    double& pressure_integral) {
  MIMI_FUNC()

  w.GradAssembly(false);
  ElementResidual<true>(q_data, w, force, pressure_integral);

  w.GradAssembly(true);
  mimi::utils::Data<double> dummy_f;
  double dummy_p;
  double* grad_data = w.local_grad_.GetData();
  double* solution_data = w.CurrentSolution().GetData();
  const double* residual_data = w.local_residual_.GetData();
  const double* fd_forward_data = w.forward_residual_.GetData();
  const int n_t_dof = w.GetTDof();
  for (int i{}; i < n_t_dof; ++i) {
    double& with_respect_to = *solution_data++;
    const double orig_wrt = with_respect_to;
    const double diff_step =
        (orig_wrt != 0.0) ? std::abs(orig_wrt) * 1.0e-8 : 1.0e-10;
    const double diff_step_inv = 1. / diff_step;

    with_respect_to = orig_wrt + diff_step;
    ElementResidual<false>(q_data, w, dummy_f, dummy_p);
    for (int j{}; j < n_t_dof; ++j) {
      *grad_data++ = (fd_forward_data[j] - residual_data[j]) * diff_step_inv;
    }
    with_respect_to = orig_wrt;
  }
}

void MortarContact::AddBoundaryResidual(const mfem::Vector& current_u,
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
    auto& w = work_data_[i_thread];
    w.GradAssembly(false);
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
      w.CurrentElementPressure(average_pressure_, beqd);
      if (w.IsPressureZero()) {
        continue;
      }

      // continue with assembly
      ElementData_& bed = beqd.GetElementData();
      w.SetDof(bed.n_dof);

      w.CurrentElementSolutionCopy(current_u, beqd);
      ElementResidual<true>(beqd.GetQuadData(), w, tl_force, local_pressure);

      {
        const std::lock_guard<std::mutex> lock(residual_mutex);
        residual.AddElementVector(bed.v_dofs, w.local_residual_.GetData());
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

void MortarContact::AddBoundaryResidualAndGrad(const mfem::Vector& current_u,
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
        auto& w = work_data_[i_thread];
        mimi::utils::Data<double> tl_force(dim_);
        tl_force.Fill(0.0);
        double local_pressure{};
        Vector_<ElementQuadData_>& element_quad_data_vec =
            precomputed_->GetElementQuadData(Name());
        for (int i{begin}; i < end; ++i) {
          ElementQuadData_& beqd = element_quad_data_vec[i];

          // first we check if we need to do anything - if pressure is zero,
          // nothing to do
          w.CurrentElementPressure(average_pressure_, beqd);
          if (w.IsPressureZero()) {
            continue;
          }

          ElementData_& bed = beqd.GetElementData();
          w.SetDof(bed.n_dof);

          // get current element solution as matrix
          w.CurrentElementSolutionCopy(current_u, beqd);
          ElementResidualAndGrad(beqd.GetQuadData(),
                                 w,
                                 tl_force,
                                 local_pressure);

          // push right away
          std::lock_guard<std::mutex> lock(residual_mutex);
          const auto& vdofs = bed.v_dofs;
          residual.AddElementVector(vdofs, w.local_residual_.GetData());
          double* A = grad.GetData();
          const double* local_A = w.local_grad_.GetData();
          const auto& A_ids = bed.A_ids;
          for (int k{}; k < static_cast<int>(A_ids.size()); ++k) {
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

double MortarContact::GapNorm(const mfem::Vector& test_x, const int nthreads) {
  MIMI_FUNC()

  std::mutex gap_norm;
  double gap_squared_total{};
  auto find_gap = [&](const int begin, const int end, const int i_thread) {
    auto& w = work_data_[i_thread];
    double local_g_squared_sum{};
    Vector_<ElementQuadData_>& element_quad_data_vec =
        precomputed_->GetElementQuadData(Name());
    // this loops marked boundary elements
    for (int i{begin}; i < end; ++i) {
      // get bed
      const ElementQuadData_& beqd = element_quad_data_vec[i];
      const ElementData_& bed = beqd.GetElementData();
      w.SetDof(bed.n_dof);

      mfem::DenseMatrix& current_element_x =
          w.CurrentElementSolutionCopy(test_x, beqd);

      for (const QuadData_& q : beqd.GetQuadData()) {
        // get current position and F
        current_element_x.MultTranspose(q.N, w.distance_query_.query.data());
        // query
        nearest_distance_coeff_->NearestDistance(w.distance_query_,
                                                 w.distance_results_);
        w.distance_results_.ComputeNormal<true>(); // unit normal
        const double g = w.distance_results_.NormalGap();
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

void MortarContact::BoundaryPostTimeAdvance(const mfem::Vector& converged_u) {
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

} // namespace mimi::integrators
