#pragma once

#include "mimi/integrators/materials.hpp"
#include "mimi/integrators/nonlinear_solid.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/n_thread_exe.hpp"

namespace mimi::integrators {

/// basic integrator of nonlinear solids
/// given current x coordinate (NOT displacement)
/// Computes F and passes it to material
class NonlinearViscoSolid : public NonlinearSolid {
  constexpr static const int kMaxTrueDof = 50;
  constexpr static const int kDimDim = 9;

public:
  using Base_ = NonlinearSolid;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;

  using Base_::Base_;

  struct TemporaryData : Base_::TemporaryData {
    mfem::F_dot_;

    void SetData

        using Base_::SetShape;
    using Base_::CurrentElementSolutionCopy;
  };

  /// This one needs / stores
  /// - basis(shape) function derivative (at reference)
  /// - reference to target jacobian weight (det)
  /// - target to reference jacobian (inverse of previous)
  /// - basis derivative at target
  virtual void Prepare(const int quadrature_order = -1) {
    MIMI_FUNC()

    Base_::Prepare(quadrature_order);
  }

  /// Performs quad loop with element data and temporary data
  void QuadLoop(const mfem::DenseMatrix& x,
                const mfem::DenseMatrix& v,
                const int i_thread,
                Vector_<QuadData>& q_data,
                TemporaryData& tmp,
                mfem::DenseMatrix& residual_matrix) {
    MIMI_FUNC()

    for (QuadData& q : q_data) {
      // get dx_dX = x * dN_dX
      mfem::MultAtB(x, q.dN_dX_, tmp.F_);
      mfem::MultAtB(v, q.dN_dX_, tmp.F_dot_);

      // currently we will just use PK1
      material_->EvaluatePK1(tmp.F_,
                             tmp.F_dot_,
                             i_thread,
                             q.material_state_,
                             tmp.stress_);
      mfem::AddMult_a_ABt(q.integration_weight_ * q.det_dX_dxi_,
                          q.dN_dX_,
                          tmp.stress_,
                          residual_matrix);
    }
  }

  virtual void AssembleDomainResidual(const mfem::Vector& current_x,
                                      const mfem::Vector& current_v) {
    MIMI_FUNC()

    auto assemble_element_residual_and_maybe_grad
  }
};

} // namespace mimi::integrators
