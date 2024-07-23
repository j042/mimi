#pragma once

#include <cmath>
#include <memory>
#include <unordered_map>

#include <mfem.hpp>

#include "mimi/materials/material_state.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/n_thread_exe.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::utils {

inline std::shared_ptr<mfem::FiniteElement>
CreateFiniteElement(int const& para_dim) {
  MIMI_FUNC()

  assert(para_dim > 1);

  if (para_dim == 2) {
    return std::make_shared<mfem::NURBS2DFiniteElement>(-1);
  } else {
    return std::make_shared<mfem::NURBS3DFiniteElement>(-1);
  }
}

inline std::shared_ptr<mfem::FiniteElement>
CreateFiniteFaceElement(int const& para_dim) {
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

inline std::shared_ptr<mfem::FaceElementTransformations>
CreateFaceTransformation() {
  MIMI_FUNC()

  return std::make_shared<mfem::FaceElementTransformations>();
}

struct QuadDataExt {};

struct QuadData {
  double integration_weight;
  double det_dX_dxi;
  mfem::Vector N;
  /// basis derivative - this is usally relevant to boundary (dof x dim)
  mfem::DenseMatrix dN_dxi;
  /// basis  this is usually relevant to domain (dof x dim)
  mfem::DenseMatrix dN_dX;

  std::shared_ptr<mimi::materials::MaterialState> material_state;
  std::shared_ptr<QuadDataExt> ext;
};

struct ElementData {
  int geometry_type;
  int n_dof;
  int n_tdof;
  int i_thread;
  int id;
  bool is_bdr;

  mfem::Array<int> dofs;
  mfem::Array<int> v_dofs;
  mimi::utils::Vector<int> A_ids; // id to matrix entry

  std::shared_ptr<mfem::FiniteElement> element;

  /// in case of boundary, this may be
  /// `mimi::utils::FaceElementTransformationsExt`
  std::shared_ptr<mfem::IsoparametricTransformation> element_trans;
};

/// holds element and quad data together
struct ElementQuadData {
  int quadrature_order;
  int n_quad;

  std::shared_ptr<ElementData> element_data;
  Vector<QuadData> quad_data;

  // some optional spaces for indices. used in contact, for example.
  Vector<mfem::Array<int>> arrays;
  Vector<mfem::Vector> vectors;
  Vector<mfem::DenseMatrix> matrices;

  ElementData& GetElementData() {
    assert(element_data);
    return *element_data;
  };

  const ElementData& GetElementData() const {
    assert(element_data);
    return *element_data;
  };

  Vector<QuadData>& GetQuadData() { return quad_data; }

  const Vector<QuadData>& GetQuadData() const { return quad_data; }

  void MakeArrays(const int n) { arrays.resize(n); }

  mfem::Array<int>& GetArray(const int i) { return arrays[i]; }
  const mfem::Array<int>& GetArray(const int i) const { return arrays[i]; }

  void MakeVectors(const int n) { vectors.resize(n); }

  mfem::Vector& GetVector(const int i) { return vectors[i]; }

  void MakeMatrices(const int n) { matrices.resize(n); }

  mfem::DenseMatrix& GetMatrix(const int i) { return matrices[i]; }
  const mfem::DenseMatrix& GetMatrix(const int i) const { return matrices[i]; }
};

/// base class for precomputed data
class PrecomputedData {
public:
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;

  int n_threads_{-1};
  int n_elements_{-1};
  int n_b_elements_{-1};
  int n_v_dofs_{-1};
  int n_dofs_{-1};
  int dim_{-1};
  int v_dim_{-1};

  // duplicates to help thread safety

  /// @brief size == nthreads -- each object should have its own
  Vector_<mfem::IntegrationRules> int_rules_;

  /// @brief size == nthreads -- each object should have its own
  Vector_<const mfem::IntegrationRule*> int_rule_;

  // following can be shared as long as one thread is using it at a time

  /// @brief size == nthreads, can share as long as v_dim are the same
  Vector_<std::shared_ptr<mfem::FiniteElementSpace>> fe_spaces_;

  /// @brief size == nthreads, can share
  Vector_<std::shared_ptr<mfem::Mesh>> meshes_;

  /// @brief size == nthreads, can share as long as they are same geometry
  Vector_<std::shared_ptr<mfem::NURBSFECollection>> fe_collections_;

  /// @brief domain element data. We keep shared pointer, so that
  /// element_quad_data can take any subset if needed
  Vector<std::shared_ptr<ElementData>> domain_element_data_;

  /// @brief boundary element data. We keep shared pointer, so that
  /// element_quad_data can take any subset if needed
  Vector<std::shared_ptr<ElementData>> boundary_element_data_;

  /// @brief map containig quad data. decoupled with element data so that one
  /// element data can be associated with many different quadrature data
  std::unordered_map<std::string, Vector_<ElementQuadData>> element_quad_data_;

  // sparsity pattern for v_dofs
  std::shared_ptr<mfem::SparseMatrix> sparsity_pattern_;

  PrecomputedData() = default;
  virtual ~PrecomputedData() = default;

  Vector_<ElementQuadData>& GetElementQuadData(std::string&& name) {
    return element_quad_data_.at(name);
  }

  const Vector_<ElementQuadData>& GetElementQuadData(std::string&& name) const {
    return element_quad_data_.at(name);
  }

  void PrepareThreadSafety(mfem::FiniteElementSpace& fe_space,
                           const int nthreads);

  void PrepareElementData();

  void PrepareSparsity();

  /// given mask, we will iterate and prepare quad data. using this, you can use
  /// PrecomputeElementQuadData. Pointer here, as one can have different type of
  /// containers for masks. for nullptr input of mask, we create for all
  Vector_<ElementQuadData>& CreateElementQuadData(const std::string name,
                                                  int* mask = nullptr,
                                                  int mask_size = -1);

  Vector_<ElementQuadData>&
  CreateBoundaryElementQuadData(const std::string name,
                                int* mask = nullptr,
                                int mask_size = -1);

  void PrecomputeElementQuadData(const std::string name,
                                 const int quadrature_order,
                                 const bool dN_dX);

protected:
  void ThreadSafe() const {
    if (n_threads_ > 0 && int_rules_.size() > 0 && meshes_.size() > 0
        && fe_collections_.size() > 0 && fe_spaces_.size() > 0) {
      // all good. exit
      return;
    }
    mimi::utils::PrintAndThrowError("Please set number of threads and fespace "
                                    "with CreateThreadSafeCopies()");
  }
};

} // namespace mimi::utils
