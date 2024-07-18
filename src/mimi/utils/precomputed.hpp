#pragma once

#include <cmath>
#include <memory>
#include <unordered_map>

#include <mfem.hpp>

#include "mimi/materials/material_state.hpp"
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

  std::shared_ptr<mfem::NURBSFiniteElement> element;

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

  /// @brief size == nthreads, can share - just for boundary, we use mesh
  /// extention
  Vector_<std::shared_ptr<mimi::utils::MeshExt>> meshes_;

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
                           const int nthreads) {
    MIMI_FUNC()

    if (nthreads < 0) {
      mimi::utils::PrintAndThrowError(nthreads, " is invalid nthreads input.");
    }

    // create for each threads
    mimi::utils::PrintDebug(
        "Creating copies of IntegrationRules, Mesh, NURBSFECollection, "
        "FiniteElementSpace for each thread");
    for (int i{}; i < nthreads; ++i) {
      // create int rules
      int_rules_.emplace_back(mfem::IntegrationRules{});

      // deep copy mesh
      meshes_.emplace_back(
          std::make_shared<mimi::utils::MeshExt>(*fe_space.GetMesh(), true));

      // create fe_collection
      fe_collections_.emplace_back(
          std::make_shared<mfem::NURBSFECollection>()); // default is varing
                                                        // degrees

      // create fe spaces
      fe_spaces_.emplace_back(
          std::make_shared<mfem::FiniteElementSpace>(fe_space,
                                                     meshes_[i].get(),
                                                     fe_collections_[i].get()));
    }

    n_threads_ = nthreads;
  }

  void PrepareElementData() {
    MIMI_FUNC()

    ThreadSafe();

    mfem::FiniteElementSpace& fe_space = *fe_spaces_[0];

    n_elements_ = fe_space.GetNE();
    n_b_elements_ = fe_space.GetNBE();
    dim_ = fe_space.GetMesh()->Dimension();
    // these are global size
    n_v_dofs_ = fe_space.GetVSize();
    n_dofs_ = fe_space.GetNDofs();
    v_dim_ = fe_space.GetVDim();

    domain_element_data_.resize(n_elements_);
    boundary_element_data_.resize(n_b_elements_);

    auto process_elems =
        [&](const int begin, const int end, const int i_thread) {
          auto& mesh = *meshes_[i_thread];
          auto& fes = *fe_spaces_[i_thread];

          for (int i{begin}; i < end; ++i) {
            // get elem data
            auto& el_data_ptr = domain_element_data_[i];
            el_data_ptr = std::make_shared<ElementData>();
            auto& el_data = *el_data_ptr;
            // we will create elements and v dof trans (dof trans is used in
            // case of prolongation, but not used here.)
            // 1. create element
            el_data.element = CreatFiniteElement(dim_); // make_shared
            // 2. el trans
            el_data.element_trans = CreateTransformation(); // make_shared

            // process / fill each data
            fes.GetNURBSext()->LoadFE(i, el_data.element.get());
            fes.GetElementTransformation(i, el_data.element_trans.get());
            // however, we do need to set FE to this newly created, as it is a
            // ptr to internal obj
            el_data.element_trans->SetFE(el_data.element.get());

            // for our application there's no dof transformation.
            // if so, we can extend here
            mfem::DofTransformation* doftrans =
                fes.GetElementDofs(i, el_data.dofs);
            doftrans = fes.GetElementVDofs(i, el_data.v_dofs);

            if (doftrans) {
              mimi::utils::PrintAndThrowError(
                  "There's doftrans. There shouldn't be one according to the "
                  "documentations.");
            }

            // set properties
            el_data.geometry_type = el_data.element->GetGeomType();
            el_data.n_dof = el_data.element->GetDof();
            el_data.n_tdof = el_data.n_dof * v_dim_;
            el_data.i_thread =
                i_thread; // this is a help value, don't rely on this
            el_data.id = i;
          }
        };

    auto process_boundary_elems =
        [&](const int begin, const int end, const int i_thread) {
          auto& mesh = *meshes_[i_thread];
          auto& fes = *fe_spaces_[i_thread];
          for (int i{begin}; i < end; ++i) {
            // get bdr elem data
            auto& bel_data_ptr = boundary_element_data_[i];
            bel_data_ptr = std::make_shared<ElementData>();
            auto& bel_data = *bel_data_ptr;
            // to avoid re casting, we assign boundary element in a separate
            // variable
            auto b_element = CreatFiniteFaceElement(dim_);
            bel_data.element = b_element;
            auto b_trans = CreateFaceTransformation();
            bel_data.element_trans = b_trans;

            fes.GetNURBSext()->LoadBE(i, b_element.get());
            mesh.GetBdrFaceTransformations(i, b_trans.get());

            // we overwrite some pointers to our own copies
            // this is mask 1 - related elem
            b_trans->Elem1 =
                domain_element_data_[b_trans->Elem1No]->element_trans.get();
            // this is mask 16 - related face elem
            b_trans->SetFE(b_element.get());

            mfem::DofTransformation* doftrans =
                fes.GetBdrElementVDofs(i, bel_data.dofs);
            doftrans = fes.GetBdrElementVDofs(i, bel_data.v_dofs);

            if (doftrans) {
              mimi::utils::PrintAndThrowError(
                  "There's doftrans. There shouldn't be one according to the "
                  "documentations.");
            }

            // set properties
            bel_data.geometry_type = bel_data.element->GetGeomType();
            bel_data.n_dof = bel_data.element->GetDof();
            bel_data.n_tdof = bel_data.n_dof * v_dim_;
            bel_data.i_thread =
                i_thread; // this is a help value, don't rely on this
            bel_data.id = i;
          }
        };

    mimi::utils::NThreadExe(process_elems, n_elements_, n_threads_);
    mimi::utils::NThreadExe(process_boundary_elems, n_b_elements_, n_threads_);
  }

  void PrepareSparsity() {
    MIMI_FUNC()

    ThreadSafe();
    if (domain_element_data_.size() < 1) {
      PrepareElementData();
    }

    mfem::FiniteElementSpace& fe_space = *fe_spaces_[0];
    sparsity_pattern_ = std::make_shared<mfem::SparseMatrix>(n_v_dofs_);
    mfem::SparseMatrix& sparse = *sparsity_pattern_;

    // dummy sparsity pattern creation
    mfem::DenseMatrix dum;
    for (const auto& el_data : domain_element_data_) {
      auto& vd = el_data->v_dofs;
      dum.SetSize(vd.Size(), vd.Size());
      dum = 1.0;
      sparse.AddSubMatrix(vd, vd, dum, 0 /* (no) skip_zeros */);
    }

    // finalize / sort -> get nice A
    sparse.Finalize(0);
    sparse.SortColumnIndices();

    // fill in A
    double* a = sparse.GetData();
    const int a_size = sparse.NumNonZeroElems();
    for (int i{}; i < a_size; ++i) {
      a[i] = static_cast<double>(i);
    }

    // get direct data access ids to A
    mfem::DenseMatrix sub_mat;
    for (const auto& el_data : domain_element_data_) {
      const auto& vd = el_data->v_dofs;
      const int vd_size = vd.Size();
      const int mat_size = vd_size * vd_size;

      sub_mat.SetSize(vd_size, vd_size);
      sparse.GetSubMatrix(vd, vd, sub_mat);

      el_data->A_ids.resize(mat_size);
      const double* sm_data = sub_mat.GetData();
      for (int j{}; j < mat_size; ++j) {
        // WARNING mfem uses int, we use int, if this is a problem, we
        // have a problem
        el_data->A_ids[j] = static_cast<int>(std::lround(sm_data[j]));
      }
    }

    for (const auto& bel_data : boundary_element_data_) {
      const auto& vd = bel_data->v_dofs;
      const int vd_size = vd.Size();
      const int mat_size = vd_size * vd_size;

      sub_mat.SetSize(vd_size, vd_size);
      sparse.GetSubMatrix(vd, vd, sub_mat);

      bel_data->A_ids.resize(mat_size);
      const double* sm_data = sub_mat.GetData();
      for (int j{}; j < mat_size; ++j) {
        // WARNING mfem uses int, we use int, if this is a problem, we
        // have a problem
        bel_data->A_ids[j] = static_cast<int>(std::lround(sm_data[j]));
      }
    }
  }

  /// given mask, we will iterate and prepare quad data. using this, you can use
  /// PrecomputeElementQuadData. Pointer here, as one can have different type of
  /// containers for masks. for nullptr input of mask, we create for all
  Vector_<ElementQuadData>& CreateElementQuadData(const std::string name,
                                                  int* mask = nullptr,
                                                  int mask_size = -1) {
    MIMI_FUNC()
    Vector_<int> arange;
    if (!mask) {
      arange = Arange(0, n_elements_);
      mask = arange.data();
      mask_size = n_elements_;
    }
    assert(mask_size > 0);

    Vector_<ElementQuadData>& eq_data = element_quad_data_[name];
    eq_data.resize(mask_size);
    for (int i{}; i < mask_size; ++i) {
      ElementQuadData& eqd = eq_data[i];
      eqd.element_data = domain_element_data_[mask[i]];
    }
    return eq_data;
  }

  Vector_<ElementQuadData>&
  CreateBoundaryElementQuadData(const std::string name,
                                int* mask = nullptr,
                                int mask_size = -1) {
    MIMI_FUNC()

    Vector_<int> arange;
    if (!mask) {
      arange = Arange(0, n_b_elements_);
      mask = arange.data();
      mask_size = n_b_elements_;
    }

    Vector_<ElementQuadData>& beq_data = element_quad_data_[name];
    beq_data.resize(mask_size);
    for (int i{}; i < mask_size; ++i) {
      ElementQuadData& beqd = beq_data[i];
      beqd.element_data = boundary_element_data_[mask[i]];
    }
    return beq_data;
  }

  void PrecomputeElementQuadData(const std::string name,
                                 const int quadrature_order,
                                 const bool dN_dX) {
    MIMI_FUNC()

    Vector_<ElementQuadData>& eq_data = element_quad_data_.at(name);

    // info check
    if (eq_data.at(0).quad_data.size() > 0) {
      mimi::utils::PrintWarning("QuadData already exists for",
                                name,
                                "re-accessing");
    }

    auto precompute = [&](const int begin, const int end, const int i_thread) {
      mfem::IntegrationRules& int_rules = int_rules_[i_thread];
      mfem::DenseMatrix dxi_dX(dim_, dim_); // should be used for domain only
      for (int i{begin}; i < end; ++i) {
        ElementQuadData& eqd = eq_data[i];
        ElementData& e_data = eqd.GetElementData();
        eqd.quadrature_order = (quadrature_order < 0)
                                    ? e_data.element->GetOrder() * 2 + 3
                                    : quadrature_order;

        // prepare quadrature point loop
        const mfem::IntegrationRule& ir =
            int_rules.Get(e_data.geometry_type, eqd.quadrature_order);

        eqd.n_quad = ir.GetNPoints();
        eqd.quad_data.resize(eqd.n_quad);
        for (int j{}; j < eqd.n_quad; ++j) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(j);
          e_data.element_trans->SetIntPoint(&ip);

          QuadData& q_data = eqd.quad_data[j];
          // weights
          q_data.integration_weight = ip.weight;
          q_data.det_dX_dxi = e_data.element_trans->Weight();

          // basis
          q_data.N.SetSize(e_data.n_dof);
          e_data.element->CalcShape(ip, q_data.N);

          // basis derivative
          q_data.dN_dxi.SetSize(e_data.n_dof, e_data.element->GetDim());
          e_data.element->CalcDShape(ip, q_data.dN_dxi);

          // this is often used in
          if (dN_dX) {
            const int ref_dim = e_data.element->GetDim();
            if (ref_dim != dim_) {
              mimi::utils::PrintWarning(
                  "Non-Square Jacobian detected for dN_dX computation");
            }
            q_data.dN_dX.SetSize(e_data.n_dof, ref_dim);
            dxi_dX.SetSize(dim_, ref_dim);
            mfem::CalcInverse(e_data.element_trans->Jacobian(), dxi_dX);
            mfem::Mult(q_data.dN_dxi, dxi_dX, q_data.dN_dX);
          }
        }
      }
    };

    mimi::utils::NThreadExe(precompute,
                            static_cast<int>(eq_data.size()),
                            n_threads_);
  }

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
