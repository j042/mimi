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
/// we will keep flat iterable containers so that we can access them directly
/// using global indices. In case of pointers, we can use simple vector, else
/// use RefVector.
class PrecomputedData {
public:
  int n_threads_{1};
  int n_elem_{};
  int n_b_elem_{};
  int dim_{};

  // duplicates to help thread safety

  /// @brief size == nthreads -- each object should have its own
  PerThreadVector<std::shared_ptr<mfem::IntegrationRules>> int_rules_;

  /// @brief size == nthreads -- each object should have its own
  PerThreadVector<const mfem::IntegrationRule*> int_rule_;

  // following can be shared as long as one thread is using it at a time

  /// @brief size == nthreads, can share as long as v_dim are the same
  PerThreadVector<std::shared_ptr<mfem::FiniteElementSpace>> fe_spaces_;

  /// @brief size == nthreads, can share - just for boundary, we use mesh
  /// extention
  PerThreadVector<std::shared_ptr<mimi::utils::MeshExt>> meshes_;

  /// @brief size == nthreads, can share as long as they are same geometry
  PerThreadVector<std::shared_ptr<mfem::NURBSFECollection>> fe_collections_;

  /// @brief size == nthreads x m_elem, harmless share
  PerThreadVector<Vector<std::shared_ptr<mfem::Array<int>>>> v_dofs_;
  Vector<std::shared_ptr<mfem::Array<int>>> v_dofs_flat_;

  /// @brief size == nthreads x m_elem, harmless share
  PerThreadVector<Vector<std::shared_ptr<mfem::Array<int>>>> boundary_v_dofs_;
  Vector<std::shared_ptr<mfem::Array<int>>> boundary_v_dofs_flat_;

  /// @brief size == nthreads x m_elem, harmless share. boundary elements will
  /// also refer to this elem.
  PerThreadVector<Vector<std::shared_ptr<mfem::NURBSFiniteElement>>> elements_;
  Vector<std::shared_ptr<mfem::NURBSFiniteElement>> elements_flat_;

  /// @brief size == nthreads x m_b_elem, harmless share. boundary elements will
  /// also refer to this elem.
  PerThreadVector<Vector<std::shared_ptr<mfem::NURBSFiniteElement>>>
      boundary_elements_;
  Vector<std::shared_ptr<mfem::NURBSFiniteElement>> boundary_elements_flat_;

  /// @brief size == nthreads x m_elem, can share as long as each are separately
  /// accessed wording: target -> stress free reference -> quadrature physical
  /// -> current
  PerThreadVector<Vector<std::shared_ptr<mfem::IsoparametricTransformation>>>
      reference_to_target_element_trans_;
  Vector<std::shared_ptr<mfem::IsoparametricTransformation>>
      reference_to_target_element_trans_flat_;

  /// @brief size == n_elem, can share as long as each are separately accessed
  PerThreadVector<
      Vector<std::shared_ptr<mimi::utils::FaceElementTransformationsExt>>>
      reference_to_target_boundary_trans_;
  Vector<std::shared_ptr<mimi::utils::FaceElementTransformationsExt>>
      reference_to_target_boundary_trans_flat_;

  /// @brief
  std::unordered_map<std::string, Vector_<Vector_<double>>> scalars_;

  /// @brief size == {n_elem; n_patch} x n_quad. each integrator can have its
  /// own.
  std::unordered_map<std::string, Vector_<Vector_<mfem::Vector>>> vectors_;

  /// @brief meant to hold jacobians per whatever
  std::unordered_map<std::string, Vector_<Vector_<mfem::DenseMatrix>>>
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
        Vector_<mfem::IntegrationRules>(int_rules_.size(),
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

    // create pointer spaces for each threads
    // they all hold shared_ptr
    int_rules_.resize(n_threads_);
    int_rule_.resize(n_threads_);
    fe_spaces_.resize(n_threads_);
    meshes_.resize(n_threads_);
    fe_collections_.resize(n_threads_);

    // critical sizes - these are total sizes
    const int n_elem = fe_space.GetNE();
    const int n_b_elem = fe_space.GetNBE();
    // and dim
    const int dim = fe_space.GetMesh()->Dimension();

    // local copies
    n_elem_ = n_elem;
    n_b_elem_ = n_b_elem;
    dim_ = dim;

    // define per-thread init
    auto process_elems =
        [&](const int begin, const int end, const int i_thread) {
          // now, each thread creates its own instances
          int_rules_[i_thread] = std::make_shared<mfem::IntegrationRules>{};
          meshes_[i_thread] =
              std::make_shared<mimi::utils::MeshExt>(*fe_space.GetMesh(),
                                                     true); // deep copy mesh
          fe_collections_[i_thread] =
              std::make_shared<mfem::NURBSFECollection>(); // default is varing
                                                           // degrees
          // create fe spaces
          fe_spaces_[i_thread] = std::make_shared<mfem::FiniteElementSpace>(
              fe_space,
              meshes_[i_thread].get(),
              fe_collections_[i_thread].get());

          // deref for following parts
          auto& mesh = *meshes_[i_thread];
          auto& fes = *fe_spaces_[i_thread];

          // we need to know workload for each thread
          const int m_elem = end - begin;
          // deref thread locals
          auto& v_dofs = v_dofs_[i_thread];
          auto& elements = elements_[i_thread];
          auto& dX_dxi = reference_to_target_element_trans_[i_thread];
          // now, alloc
          v_dofs.resize(m_elem);
          elements.resize(m_elem);
          dX_dxi.resize(m_elem);
          // loop changes to local indices
          // but let's keep global indices - we use g
          for (int i{}, g{begin}; i < m_elem; ++i, ++g) {
            // is there a dof trans? I am pretty sure not
            // if so, we can extend here
            auto& v_dof = v_dofs[i];
            v_dof = std::make_shared<mfem::Array<int>>();
            mfem::DofTransformation* doftrans = fes.GetElementVDofs(g, *v_dof);
            if (doftrans) {
              mimi::utils::PrintAndThrowError(
                  "There's doftrans. There shouldn't be one according to the "
                  "documentations.");
            }

            // we will create elements and v dof trans (dof trans is used in
            // case of prolongation, but not used here.)
            // 1. create element
            auto& elem = elements[i];
            elem = CreatFiniteElement(dim); // make_shared

            auto& e_tr = dX_dxi[i];
            e_tr = CreateTransformation(); // make_shared

            // process/set FE
            fes.GetNURBSext()->LoadFE(g, elem.get());

            // prepare transformation - we could just copy paste the code, and
            // this will save GetElementVDofs, but let's not go too crazy
            fes.GetElementTransformation(g, e_tr.get());
            // however, we do need to set FE to this newly created, as it is a
            // ptr to internal obj
            e_tr->SetFE(elem.get());
          }
        };

    // define per-thread init
    auto process_boundary_elems =
        [&](const int begin, const int end, const int i_thread) {
          // deref basics
          auto& mesh = *meshes_[i_thread];
          auto& fes = *fe_spaces_[i_thread];

          // we need to know workload for each thread
          const int m_b_elem = end - begin;
          // deref thread locals
          auto& boundary_v_dofs = boundary_v_dofs_[i_thread];
          auto& boundary_elements = boundary_elements_[i_thread];
          auto& dX_dxi = reference_to_target_boundary_trans_[i_thread];
          // then alloc
          boundary_v_dofs.resize(m_b_elem);
          boundary_elements.resize(m_b_elem);
          dX_dxi.resize(m_b_elem);
          // keep global indices
          for (int i{}, g{begin}; i < m_b_elem; ++i, ++g) {
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

            // create element,
            auto& b_el = boundary_elements_[i];
            b_el = CreatFiniteFaceElement(dim);
            // get bdr element
            fes.GetNURBSext()->LoadBE(g, b_el.get());

            auto& b_tr = dX_dxi[i];
            b_tr = CreateFaceTransformation();

            // this is extended function mainly to
            mesh.GetBdrFaceTransformations(g, b_tr.get());

            // we overwrite some pointers to our own copies
            // this is mask 1 - related elem
            // set
            b_tr->Elem1 =
                reference_to_target_element_trans_flat_[b_tr->Elem1No].get();

            // this is mask 16 - related face elem
            // we need to create b_elem of our own
            fes.GetNURBSext()->LoadBE(g, b_el.get());
            b_tr->SetFE(b_el.get());
          }
        };

    // allocate vectors for each thread
    v_dofs_.resize(n_threads_);
    elements_.resize(n_threads_);
    reference_to_target_element_trans_.resize(n_threads_);
    mimi::utils::NThreadExe(process_elems, n_elem, nthreads);
    // make flat
    // this is not most efficient, but much less code
    MakeFlat2(v_dofs_, v_dofs_flat_, n_elem);
    MakeFlat2(elements_, elements_flat_, n_elem);
    MakeFlat2(reference_to_target_element_trans_,
              reference_to_target_element_trans_flat_,
              n_elem);

    // now for boundaries
    boundary_v_dofs_.resize(n_threads_);
    boundary_elements_.resize(n_threads_);
    reference_to_target_boundary_trans_.resize(n_threads_);
    mimi::utils::NThreadExe(process_boundary_elems, n_b_elem, nthreads);
    MakeFlat2(boundary_v_dofs_, boundary_v_dofs_flat_, n_b_elem);
    MakeFlat2(boundary_elements_, boundary_elements_flat_, n_b_elem);
    MakeFlat2(reference_to_target_boundary_trans_,
              reference_to_target_boundary_trans_flat_,
              n_b_elem);
  }
};

class PrecomputedElementData : public PrecomputedData {
public:
  using Base_ = PrecomputedData;
  using Base_::Base_;
};

} // namespace mimi::utils
