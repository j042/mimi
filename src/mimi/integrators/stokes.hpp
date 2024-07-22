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
      mfem::DenseMatrix element_A00, element_A01;
      mfem::Vector& thread_local_A = thread_local_As[i_thread];
      const Vector_<BlockSparseEntries>& block_sparse_entries = precomputed_->GetBlockSparseEntries();
      for (int i{begin}; i < end; ++i) { // element loop
        ElementQuadData_& v_eqd = v_elem_quad_data_vec[i];
        ElementQuadData_& p_eqd = p_elem_quad_data_vec[i];
        ElementData_& v_ed = v_eqd.GetElementData();
        ElementData_& p_ed = p_eqd.GetElementData();
        Vector_<QuadData_>& v_qd_vec = v_eqd.GetQuadData();
        Vector_<QuadData_>& p_qd_vec = p_eqd.GetQuadData();
        const BlockSparseEntries& bse = block_sparse_entries[i];

        // quad loop
        for (int q{}; q < v_qd.size(); ++q) {
          QuadData_ v_qd = v_qd_vec[q];
          QuadData_ p_qd = p_qd_vec[q];

          double mu = material_.viscosity_;
          double int_weight_vel = v_qd.integration_weight;
          double det_J_vel = v_qd.det_dX_dxi;
          mfem::DenseMatrix vel_dxi = v_qd.dN_dxi;
          
          //∫μ ∇v:∇w dΩ

          // A00: ∫μ (∇v)ᵀ:∇w dΩ
          mfem::AddMult_a_AAt(
            mu * int_weight_vel,
            vel_dxi,
            element_A00
          );

          // ∫p∇⋅w dΩ
          mfem::Vector& vel_div;
          vel_dxi.GradToDiv(vel_div);
          mfem::AddMult_a_VWt(
            int_weight_vel,
            p_qd.N,
            vel_div,
            element_A01
          );
        }

        // push to thread local A
      }
    };

    // push to global matrix - see nonlinear base
  }
  virtual void AddDomainResidual(const mfem::BlockVector& current_sol,
                                 mfem::Vector& residual) {}
  virtual void AddDomainResidualAndGrad(const mfem::BlockVector& current_sol,
                                        const double grad_factor,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) {}
};

} // namespace mimi::integrators
