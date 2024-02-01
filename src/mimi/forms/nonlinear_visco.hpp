#pragma once

#include <mfem.hpp>

#include "mimi/forms/nonlinear.hpp"
#include "mimi/integrators/nonlinear_visco_solid.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::forms {
/* Extends Mult() to incorporate mimi::integrators::nonloinear_base */
class NonlinearVisco : public Nonlinear {
public:
  using Base_ = Nonlinear;
  using NFVIPointer_ = std::shared_ptr<mimi::integrators::NonlinearViscoSolid>;

  mimi::utils::Vector<NFVIPointer_> domain_nfvi_{};
  mimi::utils::Vector<NFVIPointer_> boundary_face_nfvi_{};
  mimi::utils::Vector<const mfem::Array<int>*> boundary_markers_{};

  /// same ctor as base
  using Base_::Base_;

  virtual void AssembleGradOn() {
    MIMI_FUNC()
    for (auto& dnfi : domain_nfvi_) {
      dnfi->assemble_grad_ = true;
    }
    for (auto& bnfi : boundary_face_nfvi_) {
      bnfi->assemble_grad_ = true;
    }
  }

  virtual void AssembleGradOff() {
    MIMI_FUNC()
    for (auto& dnfi : domain_nfvi_) {
      dnfi->assemble_grad_ = false;
    }
    for (auto& bnfi : boundary_face_nfvi_) {
      bnfi->assemble_grad_ = false;
    }
  }

  virtual void FreezeStates() {
    MIMI_FUNC()

    for (auto& dnfi : domain_nfvi_) {
      dnfi->frozen_state_ = true;
    }
    for (auto& bnfi : boundary_face_nfvi_) {
      bnfi->frozen_state_ = true;
    }
  }

  virtual void MeltStates() {
    MIMI_FUNC()

    for (auto& dnfi : domain_nfvi_) {
      dnfi->frozen_state_ = false;
    }
    for (auto& bnfi : boundary_face_nfvi_) {
      bnfi->frozen_state_ = false;
    }
  }

  /// we skip a lot of checks that's performed by base here
  virtual void Mult(const mfem::Vector& current_x,
                    const mfem::Vector& current_v,
                    mfem::Vector& residual) const {
    MIMI_FUNC()

    residual = 0.0;

    AddMult(current_x, current_v, residual);
  }

  virtual void AddMult(const mfem::Vector& current_x,
                       const mfem::Vector& current_v,
                       mfem::Vector& residual) const {
    MIMI_FUNC()

    // we assemble all first - these will call nthreadexe
    // domain
    for (auto& domain_integ : domain_nfvi_) {
      domain_integ->dt_ = dt_;
      domain_integ->first_effective_dt_ = first_effective_dt_;
      domain_integ->second_effective_dt_ = second_effective_dt_;
      domain_integ->AssembleDomainResidual(current_x, current_v);

      // add to global
      const auto& el_vecs = *domain_integ->element_vectors_;
      const auto& el_vdofs = domain_integ->precomputed_->v_dofs_;

      for (int i{}; i < el_vecs.size(); ++i) {
        residual.AddElementVector(*el_vdofs[i], el_vecs[i]);
      }
    }

    // boundary
    for (auto& boundary_integ : boundary_face_nfvi_) {
      boundary_integ->dt_ = dt_;
      boundary_integ->first_effective_dt_ = first_effective_dt_;
      boundary_integ->second_effective_dt_ = second_effective_dt_;
      boundary_integ->AssembleBoundaryResidual(current_x, current_v);

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

  virtual mfem::Operator& GetGradient(const mfem::Vector& current_x,
                                      const mfem::Vector& current_v) const {
    MIMI_FUNC();

    if (Grad == NULL) {
      Base_::Grad = new mfem::SparseMatrix(Base_::fes->GetVSize());
    } else {
      *Base_::Grad = 0.0;
    }

    // we assemble all first - these will call nthreadexe
    // domain
    for (auto& domain_integ : domain_nfvi_) {
      domain_integ->dt_ = dt_;
      domain_integ->first_effective_dt_ = first_effective_dt_;
      domain_integ->second_effective_dt_ = second_effective_dt_;
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
    for (auto& boundary_integ : boundary_face_nfvi_) {
      boundary_integ->dt_ = dt_;
      boundary_integ->first_effective_dt_ = first_effective_dt_;
      boundary_integ->second_effective_dt_ = second_effective_dt_;
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

    return *Base_::Grad;
  }

  void AddDomainIntegrator(const NFVIPointer_& nlfi) {
    MIMI_FUNC()

    domain_nfvi_.push_back(nlfi);
  }

  void AddBdrFaceIntegrator(const NFVIPointer_& nlfi,
                            const mfem::Array<int>* bdr_marker = nullptr) {
    MIMI_FUNC()

    boundary_face_nfvi_.push_back(nlfi);
    boundary_markers_.push_back(bdr_marker);
  }

  void SetEssentialTrueDofs(const mfem::Array<int>& ess_true_dof_list) {
    MIMI_FUNC()

    Base_::SetEssentialTrueDofs(ess_true_dof_list);
  }
};

} // namespace mimi::forms
