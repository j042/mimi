#pragma once

#include <memory>
#include <unordered_map>

#include <mfem.hpp>

#include "mimi/utils/containers.hpp"
#include "mimi/utils/mfem_ext.hpp"
#include "mimi/utils/n_thread_exe.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::utils {

inline std::shared_ptr<mfem::NURBSFiniteElement>
CreatFiniteElement(int const& para_dim) {
  MIMI_FUNC()

  assert(para_dim > 1);

  if (para_dim == 2) {
    return std::make_shared<mfem::NURBS2DFiniteElement>(-1);
  } else {
    return std::make_shared<mfem::NURBS3DFiniteElement>(-1);
  }
}

inline std::shared_ptr<mfem::NURBSFiniteElement>
CreatFiniteFaceElement(int const& para_dim) {
  MIMI_FUNC()

  assert(para_dim > 1);

  if (para_dim == 2) {
    return std::make_shared<mfem::NURBS1DFiniteElement>(-1);
  } else {
    return std::make_shared<mfem::NURBS2DFiniteElement>(-1);
  }
}

inline std::shared_ptr<mfem::IsoparametricTransformation>
CreateTransformation() {
  MIMI_FUNC()

  return std::make_shared<mfem::IsoparametricTransformation>();
}

inline std::shared_ptr<mimi::utils::FaceElementTransformationsExt>
CreateFaceTransformation() {
  MIMI_FUNC()

  return std::make_shared<mimi::utils::FaceElementTransformationsExt>();
}

/// base class for precomputed data
class PrecomputedData {
public:
  int n_threads_{1};

  // duplicates to help thread safety

  /// @brief size == nthreads -- each object should have its own
  mimi::utils::Vector<mfem::IntegrationRules> int_rules_;

  /// @brief size == nthreads -- each object should have its own
  mimi::utils::Vector<const mfem::IntegrationRule*> int_rule_;

  // following can be shared as long as one thread is using it at a time

  /// @brief size == nthreads, can share as long as v_dim are the same
  mimi::utils::Vector<std::shared_ptr<mfem::FiniteElementSpace>> fe_spaces_;

  /// @brief size == nthreads, can share - just for boundary, we use mesh
  /// extention
  mimi::utils::Vector<std::shared_ptr<mimi::utils::MeshExt>> meshes_;

  /// @brief size == nthreads, can share as long as they are same geometry
  mimi::utils::Vector<std::shared_ptr<mfem::NURBSFECollection>> fe_collections_;

  /// @brief size == n_elem, harmless share
  mimi::utils::Vector<std::shared_ptr<mfem::Array<int>>> v_dofs_;

  /// @brief size == n_elem, harmless share
  mimi::utils::Vector<std::shared_ptr<mfem::Array<int>>> boundary_v_dofs_;

  /// @brief size == n_elem, harmless share. boundary elements will also refer
  /// to this elem.
  mimi::utils::Vector<std::shared_ptr<mfem::NURBSFiniteElement>> elements_;

  /// @brief size == n_b_elem, harmless share. boundary elements will also refer
  /// to this elem.
  mimi::utils::Vector<std::shared_ptr<mfem::NURBSFiniteElement>>
      boundary_elements_;

  /// @brief size == n_elem, can share as long as each are separately accessed
  /// wording:
  /// target -> stress free
  /// reference -> quadrature
  /// physical -> current
  mimi::utils::Vector<std::shared_ptr<mfem::IsoparametricTransformation>>
      reference_to_target_element_trans_;

  /// @brief size == n_elem, can share as long as each are separately accessed
  mimi::utils::Vector<
      std::shared_ptr<mimi::utils::FaceElementTransformationsExt>>
      reference_to_target_boundary_trans_;

  /// @brief
  std::unordered_map<std::string,
                     mimi::utils::Vector<mimi::utils::Vector<double>>>
      scalars_;

  /// @brief size == {n_elem; n_patch} x n_quad. each integrator can have its
  /// own.
  std::unordered_map<std::string,
                     mimi::utils::Vector<mimi::utils::Vector<mfem::Vector>>>
      vectors_;

  /// @brief meant to hold jacobians per whatever
  std::unordered_map<
      std::string,
      mimi::utils::Vector<mimi::utils::Vector<mfem::DenseMatrix>>>
      matrices_;

  /// @brief relevant markers for nonlinear boundary integrations
  /// this is integrator dependant
  std::unordered_map<std::string, std::shared_ptr<mfem::Array<int>>>
      boundary_attr_markers_;

  PrecomputedData() = default;
  virtual ~PrecomputedData() = default;

  virtual void Clear() {
    int_rules_.clear();
    int_rule_.clear();
    fe_spaces_.clear();
    meshes_.clear();
    fe_collections_.clear();
    v_dofs_.clear();
    elements_.clear();
    reference_to_target_element_trans_.clear();
    reference_to_target_boundary_trans_.clear();

    scalars_.clear();
    vectors_.clear();
    matrices_.clear();

    boundary_attr_markers_.clear();

    n_threads_ = -1;
  }

  /// @brief pastes shareable properties
  virtual void PasteCommonTo(std::shared_ptr<PrecomputedData>& other) {
    MIMI_FUNC()

    other->int_rules_ =
        mimi::utils::Vector<mfem::IntegrationRules>(int_rules_.size(),
                                                    mfem::IntegrationRules());
    other->fe_spaces_ = fe_spaces_;
    other->meshes_ = meshes_;
    other->fe_collections_ = fe_collections_;
    other->v_dofs_ = v_dofs_;
    other->boundary_v_dofs_ = boundary_v_dofs_;
    other->elements_ = elements_;
    other->boundary_elements_ = boundary_elements_;
    other->reference_to_target_element_trans_ =
        reference_to_target_element_trans_;
    other->reference_to_target_boundary_trans_ =
        reference_to_target_boundary_trans_;
  }

  virtual void Setup(const mfem::FiniteElementSpace& fe_space,
                     const int nthreads) {
    MIMI_FUNC()

    if (nthreads < 0) {
      mimi::utils::PrintAndThrowError(nthreads, " is invalid nthreads input.");
    }

    // reset first
    Clear();

    // save nthreads
    n_threads_ = nthreads;

    // create for each threads
    for (int i{}; i < nthreads; ++i) {
      // create int rules
      int_rules_.emplace_back(mfem::IntegrationRules{});

      // deep copy mesh
      meshes_.emplace_back(
          std::make_shared<mimi::utils::MeshExt>(*fe_space.GetMesh(), true));

      // create fe_collection - TODO - what do they really do?
      fe_collections_.emplace_back(
          std::make_shared<mfem::NURBSFECollection>()); // default is varing
                                                        // degrees

      // create fe spaces
      fe_spaces_.emplace_back(
          std::make_shared<mfem::FiniteElementSpace>(fe_space,
                                                     meshes_[i].get(),
                                                     fe_collections_[i].get()));
    }

    const int n_elem = fe_space.GetNE();
    const int n_b_elem = fe_space.GetNBE();
    const int dim = fe_space.GetMesh()->Dimension();

    // allocate vectors
    v_dofs_.resize(n_elem);
    boundary_v_dofs_.resize(n_b_elem);
    elements_.resize(n_elem);
    boundary_elements_.resize(n_b_elem);
    reference_to_target_element_trans_.resize(n_elem);
    reference_to_target_boundary_trans_.resize(n_b_elem);

    auto process_elems =
        [&](const int begin, const int end, const int i_thread) {
          auto& mesh = *meshes_[i_thread];
          auto& fes = *fe_spaces_[i_thread];

          for (int i{begin}; i < end; ++i) {
            // we will create elements and v dof trans (dof trans is used in
            // case of prolongation, but not used here.)
            // 1. create element
            auto& elem = elements_[i];
            elem = CreatFiniteElement(dim); // make_shared

            auto& e_tr = reference_to_target_element_trans_[i];
            e_tr = CreateTransformation(); // make_shared

            // process/set FE
            fes.GetNURBSext()->LoadFE(i, elem.get());

            // prepare transformation - we could just copy paste the code, and
            // this will save GetElementVDofs, but let's not go too crazy
            fes.GetElementTransformation(i, e_tr.get());
            // however, we do need to set FE to this newly created, as it is a
            // ptr to internal obj
            e_tr->SetFE(elem.get());

            // is there a dof trans? I am pretty sure not
            // if so, we can extend here
            auto& v_dof = v_dofs_[i];
            v_dof = std::make_shared<mfem::Array<int>>();
            mfem::DofTransformation* doftrans = fes.GetElementVDofs(i, *v_dof);

            if (doftrans) {
              mimi::utils::PrintAndThrowError(
                  "There's doftrans. There shouldn't be one according to the "
                  "documentations.");
            }
          }
        };

    auto process_boundary_elems = [&](const int begin,
                                      const int end,
                                      const int i_thread) {
      auto& mesh = *meshes_[i_thread];
      auto& fes = *fe_spaces_[i_thread];

      for (int i{begin}; i < end; ++i) {
        // create element,
        auto& b_el = boundary_elements_[i];
        b_el = CreatFiniteFaceElement(dim);
        // get bdr element
        fes.GetNURBSext()->LoadBE(i, b_el.get());

        auto& b_tr = reference_to_target_boundary_trans_[i];
        b_tr = CreateFaceTransformation();

        // this is extended function mainly to
        mesh.GetBdrFaceTransformations(i, b_tr.get());

        // we overwrite some pointers to our own copies
        // this is mask 1 - related elem
        // set
        b_tr->Elem1 = reference_to_target_element_trans_[b_tr->Elem1No].get();

        // this is mask 16 - related face elem
        // we need to create b_elem of our own
        fes.GetNURBSext()->LoadBE(i, b_el.get());
        b_tr->SetFE(b_el.get());

        // is there a dof trans? I am pretty sure not
        // if so, we can extend here
        auto& boundary_v_dof = boundary_v_dofs_[i];
        boundary_v_dof = std::make_shared<mfem::Array<int>>();
        mfem::DofTransformation* doftrans =
            fes.GetBdrElementVDofs(i, *boundary_v_dof);

        if (doftrans) {
          mimi::utils::PrintAndThrowError(
              "There's doftrans. There shouldn't be one according to the "
              "documentations.");
        }
      }
    };

    mimi::utils::NThreadExe(process_elems, n_elem, nthreads);

    mimi::utils::NThreadExe(process_boundary_elems, n_b_elem, nthreads);
  }
};

class PrecomputedElementData : public PrecomputedData {
public:
  using Base_ = PrecomputedData;

  using Base_::Base_;
};

} // namespace mimi::utils
