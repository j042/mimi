#include "mimi/integrators/nonlinear_solid.hpp"

namespace mimi::integrators {

void NonlinearSolid::PrepareNonlinearSolidWorkDataAndMaterial() {
  MIMI_FUNC()

  assert(material_);

  // create states if needed
  if (has_states_) {
    Vector_<ElementQuadData_>& element_quad_data_vec =
        precomputed_->GetElementQuadData("domain");
    for (ElementQuadData_& eqd : element_quad_data_vec) {
      for (QuadData_& qd : eqd.GetQuadData()) {
        qd.material_state = material_->CreateState();
      }
    }
  }

  // allocate temporary data
  work_data_.resize(n_threads_);
  for (auto& td : work_data_) {
    td.SetDim(precomputed_->dim_);
    material_->AllocateAux(td);
  }
}

void NonlinearSolid::Prepare() {
  MIMI_FUNC()

  n_elements_ = precomputed_->n_elements_;
  n_threads_ = precomputed_->n_threads_;

  // setup material
  material_->Setup(precomputed_->dim_);
  // set flag for this integrator
  if (material_->CreateState()) {
    // not a nullptr -> has states
    has_states_ = true;
  } else {
    has_states_ = false;
  }

  PrepareNonlinearSolidWorkDataAndMaterial();
}

void NonlinearSolid::ElementResidualAndGrad(const Vector_<QuadData_>& q_data,
                                            NonlinearSolidWorkData& w) const {
  MIMI_FUNC()

  w.GradAssembly(false);
  ElementResidual(q_data, w);

  // now, forward fd
  w.GradAssembly(true);
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
    ElementResidual(q_data, w);
    for (int j{}; j < n_t_dof; ++j) {
      *grad_data++ = (fd_forward_data[j] - residual_data[j]) * diff_step_inv;
    }
    with_respect_to = orig_wrt;
  }
}

void NonlinearSolid::ThreadLocalResidual(const mfem::Vector& current_u) const {
  auto thread_local_residual = [&](const int begin,
                                   const int end,
                                   const int i_thread) {
    // deref relavant resources
    NonlinearSolidWorkData& w = work_data_[i_thread];
    Vector_<ElementQuadData_>& element_quad_data_vec =
        precomputed_->GetElementQuadData("domain");

    // initialize thread local residual
    w.thread_local_residual_.SetSize(current_u.Size());
    w.thread_local_residual_ = 0.0;
    w.GradAssembly(false);

    // assemble to thread local
    for (int i{begin}; i < end; ++i) {
      const ElementQuadData_& eqd = element_quad_data_vec[i];
      const ElementData_& ed = eqd.GetElementData();
      w.SetDof(ed.n_dof);
      w.CurrentElementSolutionCopy(current_u, ed.v_dofs);
      ElementResidual<false>(eqd.GetQuadData(), w);
      w.thread_local_residual_.AddElementVector(ed.v_dofs,
                                                w.local_residual_.GetData());
    }
  };

  mimi::utils::NThreadExe(thread_local_residual, n_elements_, n_threads_);
}

void NonlinearSolid::ThreadLocalResidualAndGrad(const mfem::Vector& current_u,
                                                const int A_nnz) const {
  auto thread_local_residual_and_grad = [&](const int begin,
                                            const int end,
                                            const int i_thread) {
    // deref relavant resources
    NonlinearSolidWorkData& w = work_data_[i_thread];
    Vector_<ElementQuadData_>& element_quad_data_vec =
        precomputed_->GetElementQuadData("domain");

    // initialize thread local residual
    w.thread_local_residual_.SetSize(current_u.Size());
    w.thread_local_residual_ = 0.0;
    w.thread_local_A_.SetSize(A_nnz);
    w.thread_local_A_ = 0.0;

    w.GradAssembly(false);

    // assemble to thread local
    for (int i{begin}; i < end; ++i) {
      // prepare assembly
      const ElementQuadData_& eqd = element_quad_data_vec[i];
      const ElementData_& ed = eqd.GetElementData();
      w.SetDof(ed.n_dof);
      w.CurrentElementSolutionCopy(current_u, ed.v_dofs);

      // assemble
      ElementResidualAndGrad(eqd.GetQuadData(), w);
      // push
      w.thread_local_residual_.AddElementVector(ed.v_dofs,
                                                w.local_residual_.GetData());
      double* A = w.thread_local_A_.GetData();
      const double* local_A = w.local_grad_.GetData();
      for (const int a_id : ed.A_ids) {
        A[a_id] += *local_A++;
      }
    }
  };

  mimi::utils::NThreadExe(thread_local_residual_and_grad,
                          n_elements_,
                          n_threads_);
}

void NonlinearSolid::AddDomainResidual(const mfem::Vector& current_u,
                                       mfem::Vector& residual) const {
  // pass time stepping info to material
  material_->dt_ = dt_;
  material_->first_effective_dt_ = first_effective_dt_;
  material_->second_effective_dt_ = second_effective_dt_;

  ThreadLocalResidual(current_u);
  AddThreadLocalResidual(work_data_, n_threads_, residual);
}

void NonlinearSolid::AddDomainResidualAndGrad(const mfem::Vector& current_u,
                                              const double grad_factor,
                                              mfem::Vector& residual,
                                              mfem::SparseMatrix& grad) const {

  material_->dt_ = dt_;
  material_->first_effective_dt_ = first_effective_dt_;
  material_->second_effective_dt_ = second_effective_dt_;

  ThreadLocalResidualAndGrad(current_u, grad.NumNonZeroElems());
  AddThreadLocalResidualAndGrad(work_data_,
                                n_threads_,
                                residual,
                                grad_factor,
                                grad);
}

void NonlinearSolid::DomainPostTimeAdvance(const mfem::Vector& converged_x) {
  MIMI_FUNC()

  if (!has_states_)
    return;
  auto accumulate_states =
      [&](const int begin, const int end, const int i_thread) {
        // deref relavant resources
        NonlinearSolidWorkData& w = work_data_[i_thread];
        Vector_<ElementQuadData_>& element_quad_data_vec =
            precomputed_->GetElementQuadData("domain");
        for (int i{begin}; i < end; ++i) {
          ElementQuadData_& eqd = element_quad_data_vec[i];
          const ElementData_& ed = eqd.GetElementData();
          w.SetDof(ed.n_dof);
          w.CurrentElementSolutionCopy(converged_x, ed.v_dofs);
          ElementResidual<true>(eqd.GetQuadData(), w);
        }
      };
  mimi::utils::NThreadExe(accumulate_states, n_elements_, n_threads_);
}

} // namespace mimi::integrators
