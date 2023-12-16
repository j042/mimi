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

  virtual void LineSearchOn() {
    MIMI_FUNC()
    for (auto& dnfi : domain_nfi_) {
      dnfi->line_search_assembly_ = true;
    }
    for (auto& bnfi : boundary_face_nfi_) {
      bnfi->line_search_assembly_ = true;
    }
  }

  virtual void LineSearchOff() {
    MIMI_FUNC()
    for (auto& dnfi : domain_nfi_) {
      dnfi->line_search_assembly_ = false;
    }
    for (auto& bnfi : boundary_face_nfi_) {
      bnfi->line_search_assembly_ = false;
    }
  }

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
        residual.AddElementVector(*el_vdofs[i], el_vecs[i]);
      }
    }

    // boundary
    for (auto& boundary_integ : boundary_face_nfi_) {
      boundary_integ->AssembleBoundaryResidual(current_x);

      // add to global
      const auto& bel_vecs = *boundary_integ->boundary_element_vectors_;
      const auto& bel_vdofs = boundary_integ->precomputed_->boundary_v_dofs_;
      for (const auto& boundary_marks :
           boundary_integ->marked_boundary_elements_) {
        residual.AddElementVector(*bel_vdofs[boundary_marks],
                                  bel_vecs[boundary_marks]);
      }
    }

    // set true dofs - if we have time, we could use nthread this.
    for (const auto& tdof : Base_::ess_tdof_list) {
      residual[tdof] = 0.0;
    }
  }

  virtual mfem::Operator& GetGradient(const mfem::Vector& current_x) const {
    MIMI_FUNC();

    if (Grad == NULL) {
      Base_::Grad = new mfem::SparseMatrix(Base_::fes->GetVSize());
    } else {
      *Base_::Grad = 0.0;
    }

    // we assemble all first - these will call nthreadexe
    // domain
    for (auto& domain_integ : domain_nfi_) {
      domain_integ->AssembleDomainGrad(current_x);

      // add to global
      const auto& el_mats = *domain_integ->element_matrices_;
      const auto& el_vdofs = domain_integ->precomputed_->v_dofs_;

      for (int i{}; i < el_mats.size(); ++i) {
        const auto& vdofs = *el_vdofs[i];
        Base_::Grad->AddSubMatrix(vdofs, vdofs, el_mats[i], 0 /* skip_zeros */);
      }
    }

    // boundary
    for (auto& boundary_integ : boundary_face_nfi_) {
      boundary_integ->AssembleBoundaryGrad(current_x);

      // add to global
      const auto& bel_mats = *boundary_integ->boundary_element_matrices_;
      const auto& bel_vdofs = boundary_integ->precomputed_->boundary_v_dofs_;
      for (const auto& boundary_marks :
           boundary_integ->marked_boundary_elements_) {
        const auto& b_vdofs = *bel_vdofs[boundary_marks];

        Base_::Grad->AddSubMatrix(b_vdofs,
                                  b_vdofs,
                                  bel_mats[boundary_marks],
                                  0 /* skip_zeros */);
      }
    }

    if (!Base_::Grad->Finalized()) {
      Base_::Grad->Finalize(0 /* skip_zeros */);
    }

    // set true dofs - if we have time, we could use nthread this (?)
    for (const auto& tdof : Base_::ess_tdof_list) {
      Base_::Grad->EliminateRowCol(tdof);
    }

    auto* dense = Base_::Grad->ToDenseMatrix();
    dense->PrintMatlab();
    delete dense;

    return *Base_::Grad;
  }

  void AddDomainIntegrator(const NFIPointer_& nlfi) {
    MIMI_FUNC()

    domain_nfi_.push_back(nlfi);
  }

  void AddBdrFaceIntegrator(const NFIPointer_& nlfi,
                            const mfem::Array<int>* bdr_marker = nullptr) {
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
