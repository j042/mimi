#pragma once

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/utils/containers.hpp"
#include "mimi/utils/precomputed.hpp"
#include "mimi/utils/print.hpp"
#include "mimi/utils/runtime_communication.hpp"

namespace mimi::integrators {

class NonlinearBase : public mfem::NonlinearFormIntegrator {
public:
  std::shared_ptr<mimi::utils::PrecomputedData> precomputed_;
  std::shared_ptr<mimi::utils::RuntimeCommunication> runtime_communication_;

  std::string name_;

  /// time step size, in case you need them
  /// nonlinear forms will set them
  double dt_{0.0};
  double first_effective_dt_{0.0};  // this is for x
  double second_effective_dt_{0.0}; // this is for x_dot (=v).

  /// marker for this bdr face integ
  const mfem::Array<int>* boundary_marker_ = nullptr;

  /// ids of contributing boundary elements, extracted based on boundary_marker_
  mimi::utils::Vector<int> marked_boundary_elements_;

  /// basic ctor saved ptr to precomputed and name to use as key in precomputed
  NonlinearBase(
      const std::string& name,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : name_(name),
        precomputed_(precomputed) {}

  /// Name of the integrator
  virtual std::string Name() { return name_; }
  virtual const std::string& Name() const { return name_; }

  virtual std::shared_ptr<mimi::utils::RuntimeCommunication>
  RuntimeCommunication() {
    MIMI_FUNC()
    if (!runtime_communication_) {
      mimi::utils::PrintAndThrowError("runtime_communication_ not set.");
    }
    return runtime_communication_;
  }

  /// Precompute call interface
  virtual void Prepare();

  virtual void AddDomainResidual(const mfem::Vector& current_x,
                                 mfem::Vector& residual) const;

  virtual void AddDomainGrad(const mfem::Vector& current_x,
                             mfem::SparseMatrix& grad) const;

  virtual void AddDomainResidualAndGrad(const mfem::Vector& current_x,
                                        const double grad_factor,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) const;

  virtual void DomainPostTimeAdvance(const mfem::Vector& current_x);

  virtual void AddBoundaryResidual(const mfem::Vector& current_x,
                                   mfem::Vector& residual);

  virtual void AddBoundaryGrad(const mfem::Vector& current_x,
                               mfem::SparseMatrix& grad);

  virtual void AddBoundaryResidualAndGrad(const mfem::Vector& current_x,
                                          const double grad_factor,
                                          mfem::Vector& residual,
                                          mfem::SparseMatrix& grad);

  virtual void BoundaryPostTimeAdvance(const mfem::Vector& current_x);

  virtual void SetBoundaryMarker(const mfem::Array<int>* b_marker) {
    MIMI_FUNC()

    boundary_marker_ = b_marker;
  }

  virtual mimi::utils::Vector<int>& MarkedBoundaryElements();

  template<typename TemporaryDataVector>
  static void AddThreadLocalResidual(const TemporaryDataVector& t_data_vec,
                                     const int n_threads,
                                     mfem::Vector& residual) {
    auto global_push = [&](const int begin, const int end, const int i_thread) {
      double* destination_begin = residual.GetData() + begin;
      const int size = end - begin;
      // we can make this more efficient by excatly figuring out the support
      // range. currently loop all
      for (const auto& tmp : t_data_vec) {
        const double* tl_begin = tmp.thread_local_residual_.GetData() + begin;
        for (int j{}; j < size; ++j) {
          destination_begin[j] += tl_begin[j];
        }
      }
    };
    mimi::utils::NThreadExe(global_push, residual.Size(), n_threads);
  }

  /// general elementwise reduction operation
  template<typename TemporaryDataVector>
  static void
  AddThreadLocalResidualAndGrad(const TemporaryDataVector& t_data_vec,
                                const int n_threads,
                                mfem::Vector& residual,
                                const double grad_factor,
                                mfem::SparseMatrix& grad) {

    auto global_push = [&](const int, const int, const int i_thread) {
      const int A_nnz = grad.NumNonZeroElems();
      int res_b, res_e, grad_b, grad_e;
      mimi::utils::ChunkRule(residual.Size(),
                             n_threads,
                             i_thread,
                             res_b,
                             res_e);
      mimi::utils::ChunkRule(A_nnz, n_threads, i_thread, grad_b, grad_e);

      double* destination_residual = residual.GetData() + res_b;
      const int residual_size = res_e - res_b;
      double* destination_grad = grad.GetData() + grad_b;
      const int grad_size = grad_e - grad_b;

      // we can make this more efficient by excatly figuring out the support
      // range. currently loop all
      for (const auto& tmp : t_data_vec) {
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

    // size of total is irrelevant as long as they are same/bigger than
    // n_threads_
    mimi::utils::NThreadExe(global_push, n_threads, n_threads);
  }

  virtual double GapNorm(const mfem::Vector& test_x, const int nthreads);
};

} // namespace mimi::integrators
