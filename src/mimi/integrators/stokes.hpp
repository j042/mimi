#pragma once

#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/materials/fluid_materials.hpp"
#include "mimi/utils/precomputed.hpp"

namespace mimi::integrators {
class Stokes : public NonlinearBase {
protected:
  std::shared_ptr<mimi::utils::FluidMaterialBase> material_;
  std::shared_ptr<mimi::utils::FluidPrecomputedData> precomputed_;

  std::unique_ptr<mfem::SparseMatrix> bilinear_matrix_;
  int n_elements_;
  int n_threads_;

public:
  using Base_ = NonlinearBase;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;
  using ElementQuadData_ = mimi::utils::ElementQuadData;
  using ElementData_ = mimi::utils::ElementData;
  using QuadData_ = mimi::utils::QuadData;

  Stokes(const std::string& name,
         const std::shared_ptr<mimi::utils::FluidMaterialBase> mat,
         const std::shared_ptr<mimi::utils::FluidPrecomputedData>& precomputed)
      : NonlinearBase(name, nullptr),
        material_(mat),
        precomputed_(precomputed) {}

  virtual void Prepare() {
    MIMI_FUNC()

    n_elements_ = precomputed_->GetVelocity().n_elements_;
    n_threads_ = precomputed_->GetVelocity().n_threads_;

    // copy sparsity
    bilinear_matrix_ =
        std::make_unique<mfem::SparseMatrix>(precomputed_->sparsity_pattern_);
    bilinear_matrix_ = 0.0;
    Vector_<mfem::Vector> thread_local_As;
    thread_local_As.resize(precomputed_->GetVelocity()->n_threads_);
    auto assemble_AB = [&](const int begin, const int end, const int i_thread) {
      mimi::utils::PrecomputedData& v_pre = precomputed_->GetVelocity();
      mimi::utils::PrecomputedData& p_pre = precomputed_->GetPressure();
      auto& v_elem_quad_data_vec = v_pre->GetElementQuadData("domain");
      auto& p_elem_quad_data_vec = p_pre->GetElementQuadData("domain");

      // list all the elements that's required for assembly.
      mfem::DenseMatrix element_A00, element_A01, element_A10, element_A00_aux;
      mfem::Vector& thread_local_A = thread_local_As[i_thread];
      thread_local_A.SetSize(precomputed_->sparsity_pattern_.NumNonZeroElems());
      thread_local_A = 0.0;
      const Vector_<BlockSparseEntries>& block_sparse_entries =
          precomputed_->GetBlockSparseEntries();
      // currently, we only have constant viscosity. Later, put this input quad
      // loop
      const double mu = material_.Viscosity();
      const int dim = v_pre->dim_;
      for (int i{begin}; i < end; ++i) { // element loop
        ElementQuadData_& v_eqd = v_elem_quad_data_vec[i];
        ElementQuadData_& p_eqd = p_elem_quad_data_vec[i];
        ElementData_& v_ed = v_eqd.GetElementData();
        ElementData_& p_ed = p_eqd.GetElementData();
        Vector_<QuadData_>& v_qd_vec = v_eqd.GetQuadData();
        Vector_<QuadData_>& p_qd_vec = p_eqd.GetQuadData();
        const BlockSparseEntries& bse = block_sparse_entries[i];

        // set sizes for this element and initialize
        element_A00.SetSize(v_ed.n_tdof, v_ed.n_tdof);
        element_A00_aux.SetSize(v_ed.n_dof, v_ed.n_dof);
        element_A01.SetSize(v_ed.n_tdof, p_ed.n_dof);
        element_A00 = 0.0;
        element_A01 = 0.0;

        // quad loop
        for (int q{}; q < v_qd.size(); ++q) {
          QuadData_ v_qd = v_qd_vec[q];
          QuadData_ p_qd = p_qd_vec[q];

          // A00: ∫μ ∇v dot ∇w dΩ
          mfem::Mult_a_AAt(mu * v_qd.integration_weight * v_qd.det_dX_dxi,
                           v_qd.dN_dX,
                           element_A00_aux);
          for (int d{}; d < dim; ++d) {
            element_A00.AddMatrix(element_A00_aux,
                                  v_ed.n_dof * d,
                                  v_ed.n_dof * d);
          }

          // ∫μ ∇v:∇w dΩ + ∫μ (∇v)ᵀ:∇w dΩ ?

          // ∫p∇⋅w dΩ
          mfem::Vector vel_div;
          vel_div.SetDataAndSize(v_qd.dN_dX.GetData(), v_ed.n_tdof);
          mfem::AddMult_a_VWt(v_qd.integration_weight * v_qd.det_dX_dxi,
                              vel_div,
                              p_qd.N,
                              element_A01);
        } // end quad

        // push to thread local A
        thread_local_As.AddElementVector(bse.A00_ids_, element_A00.GetData());
        thread_local_As.AddElementVector(bse.A01_ids_, element_A01.GetData());
        element_A01.Transpose(); // now this is A10;
        thread_local_As.AddElementVector(bse.A10_ids_, element_A01.GetData());
      }
    };

    mimi::utils::NThreadExe(assemble_AB, n_elements_, n_threads_);

    // reduce thread local to one global
    auto thread_reduce =
        [&](const int begin, const int end, const int i_thread) {
          double* destination_begin = bilinear_matrix_->GetData() + begin;
          const int size = end - begin;
          for (mfem::Vector& tl_A : thread_local_As) {
            const double* tl_begin = tl_A.GetData() + begin;
            for (int i{}; i < size; ++i) {
              destination_begin[i] += tl_begin[i];
            }
          }
        };
    mimi::utils::NThreadExe(thread_reduce,
                            bilinear_matrix_->NumNonZeroElems(),
                            n_threads);
  }
  virtual void AddDomainResidual(const mfem::BlockVector& current_sol,
                                 mfem::Vector& residual) {
    MIMI_FUNC()
    bilinear_matrix_->Mult(current_sol, residual);
  }
  virtual void AddDomainResidualAndGrad(const mfem::BlockVector& current_sol,
                                        const double grad_factor,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) {
    MIMI_FUNC()

    bilinear_matrix_->Mult(current_sol, residual);
    std::copy_n(bilinear_matrix_->GetData(),
                bilinear_matrix_->NumNonZeroElems(),
                grad.GetData());
  }
};

} // namespace mimi::integrators
