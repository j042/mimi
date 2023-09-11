#pragma once

#include <mfem.hpp>

#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::forms {
/* Extends Mult() to incorporate mimi::integrators::nonloinear_base */
class Nonlinear : public mfem::NonlinearForm {
public:
  using Base_ = mfem::NonlinearForm;
  using NFIPointer_ = std::shared_ptr<mimi::integrators::NonlinearBase>;

  mimi::utils::Vector<NFIPointer_> domain_nfi_{};
  mimi::utils::Vector<NFIPointer_> boundary_face_nfi_{};
  mimi::utils::Vector<const mfem::Array<int>*> boundary_markers_{};

  /// same ctor as base
  using Base_::Base_;

  /// we skip a lot of checks that's performed by base here
  virtual void Mult(const mfem::Vector& current_x,
                    mfem::Vector& residual) const {
    MIMI_FUNC()

    residual = 0.0;

    // we assemble all first - these will call nthreadexe
    // domain
    for (auto& domain_integ : domain_nfi_) {
      domain_integ->AssembleDomainResidual(current_x);

      // add to global
      const auto& el_vecs = *domain_integ->element_vectors_;
      const auto& el_vdofs = domain_integ->precomputed_->v_dofs_;
      for (int i{}; i < el_vecs.size(); ++i) {
        residual.AddElementVector(*el_vdofs[i], el_vecs[i])
      }
    }

    // boundary
    for (auto& boundary_integ : boundary_face_nfi_) {
      boundary_integ->AssembleBoundaryResidual(current_x);

      // add to global
      const auto& bel_vecs = *domain_integ->boundary_element_vectors_;
      const auto& bel_vdofs = domain_integ->precomputed_->boundary_v_dofs_;
      for (const auto& boundary_marks :
           domain_integ->marked_boundary_elements_) {
        residual.AddElementVector(*bel_vdofs[boundary_marks],
                                  bel_vecs[boundary_marks])
      }
    }

    // set true dofs - if we have time, we could use nthread this.
    for (const auto& tdof : Base_::ess_tdof_list) {
      y[tdof] = 0.0;
    }
  }

  void AddDomainIntegrator(const NFIPointer_& nlfi) {
    MIMI_FUNC()

    domain_nfi_.push_back(nlfi);
  }

  void AddBdrFaceIntegrator(const NFIPointer_& nlfi,
                            const Array<int>* bdr_marker = nullptr) {
    MIMI_FUNC()

    boundary_face_nfi_.push_back(nlfi);
    boundary_markers_.push_back(bdr_marker);
  }

  void SetEssentialTrueDofs(const mfem::Array<int>& ess_true_dof_list) {
    MIMI_FUNC()

    Base_::SetEssentialTrueDofs(ess_true_dof_list);
  }
};

} // namespace mimi::forms
