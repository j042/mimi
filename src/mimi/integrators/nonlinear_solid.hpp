#pragma once

#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/materials/materials.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/n_thread_exe.hpp"

namespace mimi::integrators {

/// basic integrator of nonlinear solids
/// given current x coordinate (NOT displacement)
/// Computes F and passes it to material
class NonlinearSolid : public NonlinearBase {
public:
  /// used to project
  std::unique_ptr<mfem::SparseMatrix> m_mat_;
  mfem::CGSolver mass_inv_;
  mfem::DSmoother mass_inv_prec_;
  mfem::UMFPackSolver mass_inv_direct_;
  mfem::Vector integrated_;
  mfem::Vector projected_; /* optional use */

  using Base_ = NonlinearBase;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;
  using ElementQuadData_ = mimi::utils::ElementQuadData;
  using ElementData_ = mimi::utils::ElementData;
  using QuadData_ = mimi::utils::QuadData;

protected:
  /// number of threads for this system
  int n_threads_;
  int n_elements_;
  mutable int A_nnz_;
  /// material related
  std::shared_ptr<mimi::materials::MaterialBase> material_;
  // no plan for mixed material, so we can have this flag in integrator level
  bool has_states_;

private:
  mutable Vector_<TemporaryData> temporary_data_;

public:
  NonlinearSolid(
      const std::string& name,
      const std::shared_ptr<mimi::materials::MaterialBase>& material,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : NonlinearBase(name, precomputed),
        material_{material} {}

  virtual const std::string& Name() const { return name_; }

  virtual void PrepareTemporaryDataAndMaterial() {
    MIMI_FUNC()

    assert(material_);

    // create states if needed
    if (has_states_) {
      Vector_<ElementQuadData_>& element_quad_data_vec =
          precomputed_->GetElementQuadData("domain");
      for (ElementQuadData_& eqd : element_quad_data_vec) {
        for (QuadData_& qd : eqd.GetQuadData()) {
          qd.material_state_ = material_->CreateState();
        }
      }
    }

    // allocate temporary data
    temporary_data_.resize(n_threads_);
    for (auto& td : temporary_data_) {
      td.SetDim(precomputed_->dim_);
      material_->AllocateAux(td);
    }
  }

  /// This one needs / stores
  /// - basis(shape) function derivative (at reference)
  /// - reference to target jacobian weight (det)
  /// - target to reference jacobian (inverse of previous)
  /// - basis derivative at target
  virtual void Prepare() {
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

    PrepareTemporaryDataAndMaterial();
  }

  /// reusable element assembly routine. Make sure CurrentElementSolutionCopy is
  /// called beforehand. Specify residual destination to enabl
  template<bool accumulate_state_only = false>
  void ElementResidual(std::conditional_t<accumulate_state_only,
                                          Vector_<QuadData_>,
                                          const Vector_<QuadData_>>& q_data,
                       TemporaryData& tmp) const {
    MIMI_FUNC()

    tmp.ResidualMatrix() = 0.0;

    // quad loop - we assume tmp.element_x_mat_ is prepared
    for (auto& q : q_data) {
      tmp.ComputeF(q.dN_dX_);
      if constexpr (!accumulate_state_only) {
        material_->EvaluatePK1(q.material_state_, tmp, tmp.stress_);
        mfem::AddMult_a_ABt(q.integration_weight_ * q.det_dX_dxi_,
                            q.dN_dX_,
                            tmp.stress_,
                            tmp.ResidualMatrix());
      } else {
        material_->Accumulate(q.material_state_, tmp);
      }
    }
  }

  /// reusable element residual and jacobian assembly. Make sure
  /// CurrentElementSolutionCopy is called beforehand
  void ElementResidualAndGrad(const Vector_<QuadData_>& q_data,
                              TemporaryData& tmp) const {
    MIMI_FUNC()

    tmp.GradAssembly(false);
    ElementResidual(q_data, tmp);

    // now, forward fd
    tmp.GradAssembly(true);
    double* grad_data = tmp.local_grad_.GetData();
    double* solution_data = tmp.CurrentSolution().GetData();
    const double* residual_data = tmp.local_residual_.GetData();
    const double* fd_forward_data = tmp.forward_residual_.GetData();
    const int n_t_dof = tmp.GetTDof();
    for (int i{}; i < n_t_dof; ++i) {
      double& with_respect_to = *solution_data++;
      const double orig_wrt = with_respect_to;
      const double diff_step = std::abs(orig_wrt) * 1.0e-8;
      // a safer option is: (orig_wrt != 0.0) ? std::abs(orig_wrt) * 1.0e-8
      // : 1.0e-8;
      const double diff_step_inv = 1. / diff_step;

      with_respect_to = orig_wrt + diff_step;
      ElementResidual(q_data, tmp);
      for (int j{}; j < n_t_dof; ++j) {
        *grad_data++ = (fd_forward_data[j] - residual_data[j]) * diff_step_inv;
      }
      with_respect_to = orig_wrt;
    }
  }

  void ThreadLocalResidual(const mfem::Vector& current_x) const {
    auto thread_local_residual =
        [&](const int begin, const int end, const int i_thread) {
          // deref relavant resources
          TemporaryData& tmp = temporary_data_[i_thread];
          Vector_<ElementQuadData_>& element_quad_data_vec =
              precomputed_->GetElementQuadData("domain");

          // initialize thread local residual
          tmp.thread_local_residual_.SetSize(current_x.Size());
          tmp.thread_local_residual_ = 0.0;
          tmp.GradAssembly(false);

          // assemble to thread local
          for (int i{begin}; i < end; ++i) {
            const ElementQuadData_& eqd = element_quad_data_vec[i];
            const ElementData_& ed = eqd.GetElementData();
            tmp.SetDof(ed.n_dof_);
            tmp.CurrentElementSolutionCopy(current_x, ed.v_dofs_);
            ElementResidual<false>(eqd.GetQuadData(), tmp);
            tmp.thread_local_residual_.AddElementVector(
                ed.v_dofs_,
                tmp.local_residual_.GetData());
          }
        };

    mimi::utils::NThreadExe(thread_local_residual, n_elements_, n_threads_);
  }

  void AddThreadLocalResidual(mfem::Vector& residual) const {
    auto global_push = [&](const int begin, const int end, const int i_thread) {
      double* destination_begin = residual.GetData() + begin;
      const int size = end - begin;
      // we can make this more efficient by excatly figuring out the support
      // range. currently loop all
      for (const TemporaryData& tmp : temporary_data_) {
        const double* tl_begin = tmp.thread_local_residual_.GetData() + begin;
        for (int j{}; j < size; ++j) {
          destination_begin[j] += tl_begin[j];
        }
      }
    };
    mimi::utils::NThreadExe(global_push, residual.Size(), n_threads_);
  }

  void ThreadLocalResidualAndGrad(const mfem::Vector& current_x) const {
    auto thread_local_residual_and_grad =
        [&](const int begin, const int end, const int i_thread) {
          // deref relavant resources
          TemporaryData& tmp = temporary_data_[i_thread];
          Vector_<ElementQuadData_>& element_quad_data_vec =
              precomputed_->GetElementQuadData("domain");

          // initialize thread local residual
          tmp.thread_local_residual_.SetSize(current_x.Size());
          tmp.thread_local_residual_ = 0.0;
          tmp.thread_local_A_.SetSize(A_nnz_);
          tmp.thread_local_A_ = 0.0;

          tmp.GradAssembly(false);

          // assemble to thread local
          for (int i{begin}; i < end; ++i) {
            // prepare assembly
            const ElementQuadData_& eqd = element_quad_data_vec[i];
            const ElementData_& ed = eqd.GetElementData();
            tmp.SetDof(ed.n_dof_);
            tmp.CurrentElementSolutionCopy(current_x, ed.v_dofs_);

            // assemble
            ElementResidualAndGrad(eqd.GetQuadData(), tmp);

            // push
            tmp.thread_local_residual_.AddElementVector(
                ed.v_dofs_,
                tmp.local_residual_.GetData());
            double* A = tmp.thread_local_A_.GetData();
            const double* local_A = tmp.local_grad_.GetData();
            for (const int a_id : ed.A_ids_) {
              A[a_id] += *local_A++;
            }
          }
        };

    mimi::utils::NThreadExe(thread_local_residual_and_grad,
                            n_elements_,
                            n_threads_);
  }

  void AddThreadLocalResidualAndGrad(mfem::Vector& residual,
                                     const double grad_factor,
                                     mfem::SparseMatrix& grad) const {

    auto global_push = [&](const int, const int, const int i_thread) {
      int res_b, res_e, grad_b, grad_e;
      mimi::utils::ChunkRule(residual.Size(),
                             n_threads_,
                             i_thread,
                             res_b,
                             res_e);
      mimi::utils::ChunkRule(A_nnz_, n_threads_, i_thread, grad_b, grad_e);

      double* destination_residual = residual.GetData() + res_b;
      const int residual_size = res_e - res_b;
      double* destination_grad = grad.GetData() + grad_b;
      const int grad_size = grad_e - grad_b;
      // we can make this more efficient by excatly figuring out the support
      // range. currently loop all
      for (const TemporaryData& tmp : temporary_data_) {
        const double* tl_res_begin =
            tmp.thread_local_residual_.GetData() + res_b;
        for (int j{}; j < residual_size; ++j) {
          destination_residual[j] += tl_res_begin[j];
        }
        const double* tl_grad_begin = tmp.thread_local_A_.GetData() + grad_b;
        for (int j{}; j < grad_size; ++j) {
          destination_grad[j] += grad_factor * tl_grad_begin[j];
        }
      }
    };
    mimi::utils::NThreadExe(global_push, residual.Size(), n_threads_);
  }

  virtual void AddDomainResidual(const mfem::Vector& current_x,
                                 mfem::Vector& residual) const {
    // pass time stepping info to material
    material_->dt_ = dt_;
    material_->first_effective_dt_ = first_effective_dt_;
    material_->second_effective_dt_ = second_effective_dt_;

    ThreadLocalResidual(current_x);
    AddThreadLocalResidual(residual);
  }

  virtual void AddDomainGrad(const mfem::Vector& current_x,
                             mfem::SparseMatrix& grad) const {

    mimi::utils::PrintAndThrowError(
        "Currently not implemented, use AddDomainResidualAndGrad");
  }

  virtual void AddDomainResidualAndGrad(const mfem::Vector& current_x,
                                        const double grad_factor,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) const {

    material_->dt_ = dt_;
    material_->first_effective_dt_ = first_effective_dt_;
    material_->second_effective_dt_ = second_effective_dt_;

    A_nnz_ = grad.NumNonZeroElems();
    ThreadLocalResidualAndGrad(current_x);
    AddThreadLocalResidualAndGrad(residual, grad_factor, grad);
  }

  virtual void DomainPostTimeAdvance(const mfem::Vector& current_x) {
    MIMI_FUNC()

    if (!has_states_)
      return;
    auto accumulate_states =
        [&](const int begin, const int end, const int i_thread) {
          // deref relavant resources
          TemporaryData& tmp = temporary_data_[i_thread];
          Vector_<ElementQuadData_>& element_quad_data_vec =
              precomputed_->GetElementQuadData("domain");
          for (int i{begin}; i < end; ++i) {
            ElementQuadData_& eqd = element_quad_data_vec[i];
            const ElementData_& ed = eqd.GetElementData();
            tmp.SetDof(ed.n_dof_);
            tmp.CurrentElementSolutionCopy(current_x, ed.v_dofs_);
            ElementResidual<true>(eqd.GetQuadData(), tmp);
          }
        };
    mimi::utils::NThreadExe(accumulate_states, n_elements_, n_threads_);
  }
};

} // namespace mimi::integrators
