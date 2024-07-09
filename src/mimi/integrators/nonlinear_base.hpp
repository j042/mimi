#pragma once

#include <cmath>

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
  virtual void Prepare() {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("Prepare not implemented");
  };

  virtual void AddDomainResidual(const mfem::Vector& current_x,
                                 mfem::Vector& residual) const {
    mimi::utils::PrintAndThrowError("AddDomainResidual not implemented");
  };

  virtual void AddDomainGrad(const mfem::Vector& current_x,
                             mfem::SparseMatrix& grad) const {
    mimi::utils::PrintAndThrowError("AddDomainGrad not implemented");
  };

  virtual void AddDomainResidualAndGrad(const mfem::Vector& current_x,
                                        const double grad_factor,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) const {
    mimi::utils::PrintAndThrowError("AddDomainResidualAndGrad not implemented");
  };

  virtual void DomainPostTimeAdvance(const mfem::Vector& current_x) {
    mimi::utils::PrintAndThrowError("DomainPostTimeAdvance not implemented");
  }

  virtual void AddBoundaryResidual(const mfem::Vector& current_x,
                                   mfem::Vector& residual) {
    mimi::utils::PrintAndThrowError("AddBoundaryResidual not implemented");
  };

  virtual void AddBoundaryGrad(const mfem::Vector& current_x,
                               mfem::SparseMatrix& grad) {
    mimi::utils::PrintAndThrowError("AddBoundaryGrad not implemented");
  };

  virtual void AddBoundaryResidualAndGrad(const mfem::Vector& current_x,
                                          const double grad_factor,
                                          mfem::Vector& residual,
                                          mfem::SparseMatrix& grad) {
    mimi::utils::PrintAndThrowError(
        "AddBoundaryResidualAndGrad not implemented");
  };

  virtual void BoundaryPostTimeAdvance(const mfem::Vector& current_x) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("BoundaryPostTimeAdvance not implemented");
  }

  virtual void SetBoundaryMarker(const mfem::Array<int>* b_marker) {
    MIMI_FUNC()

    boundary_marker_ = b_marker;
  }

  virtual mimi::utils::Vector<int>& MarkedBoundaryElements() {
    MIMI_FUNC()
    if (!boundary_marker_) {
      mimi::utils::PrintAndThrowError("boundary_marker_ not set");
    }
    if (!precomputed_) {
      mimi::utils::PrintAndThrowError("precomputed_ not set");
    }

    if (marked_boundary_elements_.size() > 0) {
      return marked_boundary_elements_;
    }

    auto& mesh = *precomputed_->meshes_[0];
    auto& b_marker = *boundary_marker_;
    marked_boundary_elements_.reserve(precomputed_->n_b_elements_);

    // loop boundary elements and check their attributes
    for (int i{}; i < precomputed_->n_b_elements_; ++i) {
      const int bdr_attr = mesh.GetBdrAttribute(i);
      if (b_marker[bdr_attr - 1] == 0) {
        continue;
      }
      marked_boundary_elements_.push_back(i);
    }
    marked_boundary_elements_.shrink_to_fit();

    return marked_boundary_elements_;
  }

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

  virtual double GapNorm(const mfem::Vector& test_x, const int nthreads) {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("GapNorm(x, nthread) not implemented for",
                                    Name());
    return 0.0;
  }
};

/// Currently, we don't split hpp and cpp, so we move nested structure to a
/// mutual place, here.
/// temporary containers required in element assembly. one for each thread
/// Stores values required for material, quad point and element data,
/// temporarily
struct TemporaryData {
  // basic info. used to compute tdofs
  int dim_;
  int n_dof_;

  // help info for thread local to global push
  int support_start_;
  int support_end_;

  // state variable
  double det_F_{};
  bool has_det_F_{false};
  bool has_F_inv_{false};

  /// flag to inform residual destination
  bool assembling_grad_{false};

  // general assembly items
  mfem::Vector element_x_;
  mfem::DenseMatrix element_x_mat_;
  mfem::DenseMatrix local_residual_;
  mfem::DenseMatrix local_grad_;
  mfem::DenseMatrix forward_residual_;

  // used for materials
  mfem::DenseMatrix stress_;
  mfem::DenseMatrix F_;
  mfem::DenseMatrix F_inv_;
  mfem::DenseMatrix F_dot_; // for visco but put it here for easier visibility

  // used in materials - materials will visit and initiate those in
  // PrepareTemporaryData
  mfem::DenseMatrix I_;
  mfem::DenseMatrix alternative_stress_;           // for conversion
  mimi::utils::Vector<mfem::Vector> aux_vec_;      // for computation
  mimi::utils::Vector<mfem::DenseMatrix> aux_mat_; // for computation

  // this is global sized local residual / grad entries
  mfem::Vector thread_local_residual_;
  mfem::Vector thread_local_A_;

  /// this can be called once at Prepare()
  void SetDim(const int dim) {
    MIMI_FUNC()
    has_det_F_ = false;
    has_F_inv_ = false;

    dim_ = dim;

    stress_.SetSize(dim, dim);
    F_.SetSize(dim, dim);
    F_inv_.SetSize(dim, dim);
    F_dot_.SetSize(dim, dim);

    I_.Diag(1., dim);
    alternative_stress_.SetSize(dim, dim);
  }

  /// this should be called at the start of every element as NDof may change
  void SetDof(const int n_dof) {
    MIMI_FUNC()

    n_dof_ = n_dof;
    element_x_.SetSize(n_dof * dim_); // will be resized in getsubvector
    element_x_mat_.UseExternalData(element_x_.GetData(), n_dof, dim_);
    forward_residual_.SetSize(n_dof, dim_);
    local_residual_.SetSize(n_dof, dim_);
    local_grad_.SetSize(n_dof * dim_, n_dof * dim_);
  }

  int GetTDof() const {
    MIMI_FUNC()

    return dim_ * n_dof_;
  }

  // computes F and resets flags
  void ComputeF(const mfem::DenseMatrix& dNdX) {
    MIMI_FUNC()
    has_det_F_ = false;
    has_F_inv_ = false;

    mfem::MultAtB(element_x_mat_, dNdX, F_);
    mimi::utils::AddDiagonal(F_.GetData(), 1.0, dim_);
  }

  mfem::DenseMatrix& FInv() {
    MIMI_FUNC()
    if (has_F_inv_) {
      return F_inv_;
    }
    mfem::CalcInverse(F_, F_inv_);

    has_F_inv_ = true;
    return F_inv_;
  }

  double DetF() {
    MIMI_FUNC()
    if (has_det_F_) {
      return det_F_;
    }

    det_F_ = F_.Det();
    has_det_F_ = true;
    return det_F_;
  }

  mfem::DenseMatrix& CurrentElementSolutionCopy(const mfem::Vector& current_all,
                                                const mfem::Array<int>& vdofs) {
    MIMI_FUNC()

    current_all.GetSubVector(vdofs, element_x_);
    return element_x_mat_;
  }

  mfem::DenseMatrix& CurrentSolution() {
    MIMI_FUNC()

    return element_x_mat_;
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
};

struct TemporaryViscoData : TemporaryData {
  using BaseTD_ = TemporaryData;

  mfem::Vector element_v_;          // v
  mfem::DenseMatrix element_v_mat_; // v as matrix

  using BaseTD_::DetF;
  using BaseTD_::FInv;

  // computes F and resets flags
  void ComputeFAndFDot(const mfem::DenseMatrix& dNdX) {
    MIMI_FUNC()
    has_det_F_ = false;
    has_F_inv_ = false;

    // MultAtBAndA1tB:
    // MultAtB together for 2 As. adapted from mfem::MultAtB

    // loop size
    const int ah = element_x_mat_.Height();
    const int aw = element_x_mat_.Width();
    const int bw = dNdX.Width();
    // A and B's data
    const double* x_d_begin = element_x_.GetData();
    const double* v_d_begin = element_v_.GetData();
    const double* dndx_d = dNdX.Data();
    // output
    double* f_d = F_.Data();
    double* fd_d = F_dot_.Data();
    for (int j{}; j < bw; ++j) {
      const double* x_d = x_d_begin;
      const double* v_d = v_d_begin;
      for (int i{}; i < aw; ++i) {
        double f_ij{}, fd_ij{};
        for (int k{}; k < ah; ++k) {
          f_ij += x_d[k] * dndx_d[k];
          fd_ij += v_d[k] * dndx_d[k];
        }
        *(f_d++) = f_ij;
        *(fd_d++) = fd_ij;
        x_d += ah;
        v_d += ah;
      }
      dndx_d += ah;
    }

    mimi::utils::AddDiagonal(F_.GetData(), 1.0, dim_);
  }

  void SetDof(const int n_dof) {
    MIMI_FUNC()

    BaseTD_::SetDof(n_dof);

    element_v_.SetSize(n_dof * dim_);
    element_v_mat_.UseExternalData(element_v_.GetData(), n_dof, dim_);
  }

  void CurrentElementSolutionCopy(const mfem::Vector& all_x,
                                  const mfem::Vector& all_v,
                                  const mfem::Array<int>& vdofs) {
    MIMI_FUNC()

    const double* all_x_data = all_x.GetData();
    const double* all_v_data = all_v.GetData();

    double* elem_x_data = element_x_.GetData();
    double* elem_v_data = element_v_.GetData();

    for (const int& vdof : vdofs) {
      *elem_x_data++ = all_x_data[vdof];
      *elem_v_data++ = all_v_data[vdof];
    }
  }
};

} // namespace mimi::integrators
