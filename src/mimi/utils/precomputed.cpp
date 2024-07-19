#include "mimi/utils/precomputed.hpp"

namespace mimi::utils {
void PrecomputedData::PrepareThreadSafety(mfem::FiniteElementSpace& fe_space,
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

void PrecomputedData::PrepareElementData() {
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

    auto process_elems = [&](const int begin,
                             const int end,
                             const int i_thread) {
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
        el_data.element = CreateFiniteElement(dim_); // make_shared
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
        mfem::DofTransformation* doftrans = fes.GetElementDofs(i, el_data.dofs);
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
        el_data.i_thread = i_thread; // this is a help value, don't rely on this
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
            auto b_element = CreateFiniteFaceElement(dim_);
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

void PrecomputeData::PrepareSparsity() {
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

Vector_<ElementQuadData>& PrecomputedData::CreateElementQuadData(const std::string name,
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
  PrecomputedData::CreateBoundaryElementQuadData(const std::string name,
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

void PrecomputedData::PrecomputeElementQuadData(const std::string name,
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
} // namespace mimi::utils