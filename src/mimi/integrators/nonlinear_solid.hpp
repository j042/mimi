#pragma once

#include "mimi/integrators/materials.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/n_thread_exe.hpp"

namespace mimi::integrators {

/// basic integrator of nonlinear solids
/// given current x coordinate (NOT displacement)
/// Computes F and passes it to material
class NonlinearSolid : public NonlinearBase {
public:
  /// used to project
  std::unique_ptr<mfem::BilinearForm> mass_;
  std::unique_ptr<mfem::SparseMatrix> m_mat_;
  mfem::CGSolver mass_inv_;
  mfem::DSmoother mass_inv_prec_;
  mfem::UMFPackSolver mass_inv_direct_;
  mfem::Vector integrated_;

  using Base_ = NonlinearBase;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;

  /// precomputed data at quad points
  struct QuadData {
    double integration_weight_;
    mfem::Vector N_; // basis - used for post processing
    mfem::DenseMatrix dN_dX_;
    double det_dX_dxi_;
    std::shared_ptr<MaterialState> material_state_;
  };

  struct ElementData {
    int quadrature_order_;
    int geometry_type_;
    int n_quad_;
    int n_dof_; // this is not true dof
    int n_tdof_;

    std::shared_ptr<mfem::Array<int>> v_dofs_;

    mfem::Array<int> scalar_v_dofs_;
    mfem::Vector
        scalar_post_process_view_; // this is (residual view / dim) sized vector
                                   // used for post processing

    Vector_<QuadData> quad_data_;

    /// pointer to element and eltrans. don't need it,
    /// but maybe for further processing or something
    std::shared_ptr<mfem::NURBSFiniteElement> element_;
    std::shared_ptr<mfem::IsoparametricTransformation> element_trans_;

    const mfem::IntegrationRule&
    GetIntRule(mfem::IntegrationRules& thread_int_rules) const {
      MIMI_FUNC()

      return thread_int_rules.Get(geometry_type_, quadrature_order_);
    }
  };

  /// temporary containers required in element assembly
  /// mfem performs some fancy checks for allocating memories.
  /// So we create one for each thread
  struct TemporaryData {
    mfem::Vector element_x_;
    mfem::DenseMatrix element_x_mat_;
    mfem::DenseMatrix stress_;
    mfem::DenseMatrix dN_dx_;
    mfem::DenseMatrix F_;
    mfem::DenseMatrix F_inv_;
    mfem::DenseMatrix forward_residual_;

    void SetShape(const int n_dof, const int dim) {
      MIMI_FUNC()
      element_x_.SetSize(n_dof * dim); // will be resized in getsubvector
      element_x_mat_.UseExternalData(element_x_.GetData(), n_dof, dim);
      stress_.SetSize(dim, dim);
      dN_dx_.SetSize(n_dof, dim);
      F_.SetSize(dim, dim);
      F_inv_.SetSize(dim, dim);
      forward_residual_.SetSize(n_dof, dim);
    }

    mfem::DenseMatrix&
    CurrentElementSolutionCopy(const mfem::Vector& current_all,
                               const ElementData& elem_data) {
      MIMI_FUNC()

      current_all.GetSubVector(*elem_data.v_dofs_, element_x_);
      return element_x_mat_;
    }
  };

protected:
  /// number of threads for this system
  int n_threads_;
  int n_elements_;
  /// material related
  std::shared_ptr<MaterialBase> material_;
  /// element data
  Vector_<ElementData> element_data_;
  // no plan for mixed material, so we can have this flag in integrator level
  bool has_states_;

  Vector_<std::unique_ptr<mfem::NURBSMeshRules>> patch_rules_;
  int patch_quadrature_order_;

public:
  NonlinearSolid(
      const std::string& name,
      const std::shared_ptr<MaterialBase>& material,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : NonlinearBase(name, precomputed),
        material_{material} {}

  virtual const std::string& Name() const { return name_; }

  virtual Vector_<ElementData>& GetElementData() { return element_data_; }
  virtual const Vector_<ElementData>& GetElementData() const {
    return element_data_;
  }

  /// This one needs / stores
  /// - basis(shape) function derivative (at reference)
  /// - reference to target jacobian weight (det)
  /// - target to reference jacobian (inverse of previous)
  /// - basis derivative at target
  virtual void Prepare(const int quadrature_order = -1) {
    MIMI_FUNC()

    // get numbers to decide loop size
    n_elements_ = precomputed_->meshes_[0]->NURBSext->GetNE();
    // n meshes should equal to nthrea config at the beginning
    n_threads_ = precomputed_->meshes_.size();

    // get dim
    dim_ = precomputed_->meshes_[0]->Dimension();

    // setup material
    material_->Setup(dim_, n_threads_);
    // set flag for this integrator
    if (material_->CreateState()) {
      // not a nullptr -> has states
      has_states_ = true;
    } else {
      has_states_ = false;
    }

    // extract element geometry type
    geometry_type_ = precomputed_->elements_[0]->GetGeomType();

    // allocate element data
    element_data_.resize(n_elements_);

    // patch-wise integration - create for each thread
    patch_rules_.resize(n_threads_);

    auto precompute_at_elements_and_quads =
        [&](const int el_begin, const int el_end, const int i_thread) {
          // thread's obj
          auto& int_rules = precomputed_->int_rules_[i_thread];
          // auto& mesh = precomputed_->meshes_[i_thread];
          // auto& patch_rule = patch_rules_[i_thread];
          // patch_rule =
          //     std::make_unique<mfem::NURBSMeshRules>(mesh->NURBSext->GetNP(),
          //     dim_);

          // const int ir_order = (quadrature_order < 0)
          //                          ? mesh->NURBSext->GetOrders().Max() * 2 +
          //                          3 : quadrature_order;
          // patch_quadrature_order_ = ir_order;
          // // loop patch and create patchrule
          // for (int p{}; p < mesh->NURBSext->GetNP(); ++p) {
          //   mfem::Array<const mfem::KnotVector*> kv(dim_);
          //   mesh->NURBSext->GetPatchKnotVectors(p, kv);
          //   std::vector<const mfem::IntegrationRule*> ir1D(dim_);
          //   const mfem::IntegrationRule* ir =
          //       &int_rules.Get(mfem::Geometry::SEGMENT, ir_order);
          //   for (int i = 0; i < dim_; ++i) {
          //     ir1D[i] = ir->ApplyToKnotIntervals(*kv[i]);
          //   }
          //   patch_rule->SetPatchRules1D(p, ir1D);
          // }
          // patch_rule->Finalize(*mesh);
          mfem::DenseMatrix dN_dxi, dxi_dX;

          // element loop
          for (int i{el_begin}; i < el_end; ++i) {
            // prepare element level data
            auto& i_el_data = element_data_[i];

            // save (shared) pointers to element and el_trans
            i_el_data.element_ = precomputed_->elements_[i];
            i_el_data.geometry_type_ = i_el_data.element_->GetGeomType();
            i_el_data.element_trans_ =
                precomputed_->reference_to_target_element_trans_[i];
            i_el_data.n_dof_ = i_el_data.element_->GetDof();
            i_el_data.v_dofs_ = precomputed_->v_dofs_[i];
            auto& v_dofs = *i_el_data.v_dofs_;

            // v_dofs are organized in xyzxyzxyz, so we just want to skip
            // through and divide them to create scalar_vdofs
            i_el_data.scalar_v_dofs_.SetSize(v_dofs.Size() / dim_);
            for (int i{}, j{}; i < v_dofs.Size(); i += dim_, ++j) {
              i_el_data.scalar_v_dofs_[j] = v_dofs[i] / dim_;
            }

            // check suitability of temp data
            // for structure simulations, tdof and vdof are the same
            const int n_tdof = i_el_data.n_dof_ * dim_;
            i_el_data.n_tdof_ = n_tdof;

            // get quad order
            i_el_data.quadrature_order_ =
                (quadrature_order < 0) ? i_el_data.element_->GetOrder() * 2 + 3
                                       : quadrature_order;

            // get int rule
            const mfem::IntegrationRule& ir = i_el_data.GetIntRule(int_rules);
            // bool delete_intrule = false;
            // const mfem::IntegrationRule& ir =
            //     patch_rule->GetElementRule(i_el_data.element_->GetElement(),
            //                                i_el_data.element_->GetPatch(),
            //                                i_el_data.element_->GetIJK(),
            //                                i_el_data.element_->KnotVectors(),
            //                                delete_intrule);

            // prepare quad loop
            i_el_data.n_quad_ = ir.GetNPoints();
            i_el_data.quad_data_.resize(i_el_data.n_quad_);

            for (int j{}; j < i_el_data.n_quad_; ++j) {
              // get int point - this is just look up within ir
              const mfem::IntegrationPoint& ip = ir.IntPoint(j);
              i_el_data.element_trans_->SetIntPoint(&ip);

              auto& q_data = i_el_data.quad_data_[j];
              q_data.integration_weight_ = ip.weight;

              dN_dxi.SetSize(i_el_data.n_dof_, dim_);
              q_data.dN_dX_.SetSize(i_el_data.n_dof_, dim_);
              dxi_dX.SetSize(dim_, dim_);
              q_data.material_state_ = material_->CreateState();
              q_data.N_.SetSize(i_el_data.n_dof_);

              i_el_data.element_->CalcShape(ip, q_data.N_);
              i_el_data.element_->CalcDShape(ip, dN_dxi);
              mfem::CalcInverse(i_el_data.element_trans_->Jacobian(), dxi_dX);
              mfem::Mult(dN_dxi, dxi_dX, q_data.dN_dX_);
              q_data.det_dX_dxi_ = i_el_data.element_trans_->Weight();
            }

            // if (delete_intrule) {
            //   delete &ir;
            // }
          }
        };

    mimi::utils::NThreadExe(precompute_at_elements_and_quads,
                            n_elements_,
                            n_threads_);
  }

  /// Performs quad loop with element data and temporary data
  void QuadLoop(const mfem::DenseMatrix& x,
                const int i_thread,
                const Vector_<QuadData>& q_data,
                TemporaryData& tmp,
                mfem::DenseMatrix& residual_matrix) const {
    MIMI_FUNC()
    for (const QuadData& q : q_data) {
      // get dx_dX = x * dN_dX
      mfem::MultAtB(x, q.dN_dX_, tmp.F_);

      // currently we will just use PK1
      material_->EvaluatePK1(tmp.F_, i_thread, q.material_state_, tmp.stress_);
      mfem::AddMult_a_ABt(q.integration_weight_ * q.det_dX_dxi_,
                          q.dN_dX_,
                          tmp.stress_,
                          residual_matrix);
    }
  }

  /// Performs quad loop with element data and temporary data
  void AccumulateStatesAtQuads(const mfem::DenseMatrix& x,
                               const int i_thread,
                               Vector_<QuadData>& q_data,
                               TemporaryData& tmp) const {
    MIMI_FUNC()
    for (QuadData& q : q_data) {
      // get dx_dX = x * dN_dX
      mfem::MultAtB(x, q.dN_dX_, tmp.F_);

      // currently we will just use PK1
      material_->Accumulate(tmp.F_, i_thread, q.material_state_);
    }
  }

  virtual void AddDomainResidual(const mfem::Vector& current_x,
                                 const int nthreads,
                                 mfem::Vector& residual) const {
    std::mutex residual_mutex;
    // lambda for nthread assemble
    auto assemble_element_residual_and_contribute = [&](const int begin,
                                                        const int end,
                                                        const int i_thread) {
      TemporaryData tmp;
      for (int i{begin}; i < end; ++i) {
        // in
        const ElementData& e = element_data_[i];
        // set shape for tmp data - first call will allocate
        tmp.SetShape(e.n_dof_, dim_);
        // variable name is misleading - this is just local residual
        // we use this container, as we already allocate this in tmp anyways
        tmp.forward_residual_ = 0.0;

        // get current element solution as matrix
        mfem::DenseMatrix& current_element_x =
            tmp.CurrentElementSolutionCopy(current_x, e);

        // assemble residual
        QuadLoop(current_element_x,
                 i_thread,
                 e.quad_data_,
                 tmp,
                 tmp.forward_residual_);

        // push right away - seems to work quite well!
        const std::lock_guard<std::mutex> lock(residual_mutex);
        residual.AddElementVector(*e.v_dofs_, tmp.forward_residual_.GetData());
      }
    };
    mimi::utils::NThreadExe(assemble_element_residual_and_contribute,
                            n_elements_,
                            (nthreads < 1) ? n_threads_ : nthreads);
  }

  virtual void AccumulateDomainStates(const mfem::Vector& current_x) {
    MIMI_FUNC()
    if (!has_states_)
      return;
    auto accumulate_states = [&](const int begin,
                                 const int end,
                                 const int i_thread) {
      TemporaryData tmp;
      for (int i{begin}; i < end; ++i) {
        // in
        ElementData& e = element_data_[i];
        // set shape for tmp data - first call will allocate
        tmp.SetShape(e.n_dof_, dim_);

        // get current element solution as matrix
        mfem::DenseMatrix& current_element_x =
            tmp.CurrentElementSolutionCopy(current_x, e);

        // accumulate
        AccumulateStatesAtQuads(current_element_x, i_thread, e.quad_data_, tmp);
      }
    };
    mimi::utils::NThreadExe(accumulate_states, n_elements_, n_threads_);
  }

  virtual void AddDomainGrad(const mfem::Vector& current_x,
                             const int nthreads,
                             mfem::SparseMatrix& grad) const {

    std::mutex residual_mutex;
    // lambda for nthread assemble
    auto assemble_element_residual_and_grad_then_contribute =
        [&](const int begin, const int end, const int i_thread) {
          TemporaryData tmp;
          mfem::Vector local_residual;
          mfem::DenseMatrix res_view;
          mfem::DenseMatrix local_grad;
          for (int i{begin}; i < end; ++i) {
            // in
            const ElementData& e = element_data_[i];
            // e.residual_view_ = 0.0;
            local_residual.SetSize(e.n_tdof_);
            local_residual = 0.0;
            res_view.UseExternalData(local_residual.GetData(), e.n_dof_, dim_);
            local_grad.SetSize(e.n_tdof_, e.n_tdof_);

            // set shape for tmp data - first call will allocate
            tmp.SetShape(e.n_dof_, dim_);

            // get current element solution as matrix
            mfem::DenseMatrix& current_element_x =
                tmp.CurrentElementSolutionCopy(current_x, e);

            // assemble residual
            QuadLoop(current_element_x, i_thread, e.quad_data_, tmp, res_view);

            double* grad_data = local_grad.GetData();
            double* solution_data = current_element_x.GetData();
            const double* residual_data = local_residual.GetData();
            const double* fd_forward_data = tmp.forward_residual_.GetData();
            for (int j{}; j < e.n_tdof_; ++j) {
              tmp.forward_residual_ = 0.0;

              double& with_respect_to = *solution_data++;
              const double orig_wrt = with_respect_to;
              const double diff_step = std::abs(orig_wrt) * 1.0e-8;
              const double diff_step_inv = 1. / diff_step;

              with_respect_to = orig_wrt + diff_step;
              QuadLoop(current_element_x,
                       i_thread,
                       e.quad_data_,
                       tmp,
                       tmp.forward_residual_);

              for (int k{}; k < e.n_tdof_; ++k) {
                *grad_data++ =
                    (fd_forward_data[k] - residual_data[k]) * diff_step_inv;
              }
              with_respect_to = orig_wrt;
            }

            // push right away
            std::lock_guard<std::mutex> lock(residual_mutex);
            const auto& vdofs = *e.v_dofs_;
            double* A = grad.GetData();
            const double* local_A = local_grad.GetData();
            const auto& A_ids = *precomputed_->domain_A_ids_[i];
            for (int k{}; k < A_ids.size(); ++k) {
              A[A_ids[k]] += *local_A++;
            }
          }
        };

    mimi::utils::NThreadExe(assemble_element_residual_and_grad_then_contribute,
                            n_elements_,
                            (nthreads < 0) ? n_threads_ : nthreads);
  };

  virtual void AddDomainResidualAndGrad(const mfem::Vector& current_x,
                                        const int nthreads,
                                        const double grad_factor,
                                        mfem::Vector& residual,
                                        mfem::SparseMatrix& grad) const {

    std::mutex residual_mutex;
    // lambda for nthread assemble
    auto assemble_element_residual_and_grad_then_contribute =
        [&](const int begin, const int end, const int i_thread) {
          TemporaryData tmp;
          mfem::DenseMatrix local_residual;
          mfem::DenseMatrix local_grad;
          for (int i{begin}; i < end; ++i) {
            // in
            const ElementData& e = element_data_[i];

            local_residual.SetSize(e.n_dof_, dim_);
            local_grad.SetSize(e.n_tdof_, e.n_tdof_);
            local_residual = 0.0;
            // local_grad = 0.0;

            // set shape for tmp data - first call will allocate
            tmp.SetShape(e.n_dof_, dim_);

            // get current element solution as matrix
            mfem::DenseMatrix& current_element_x =
                tmp.CurrentElementSolutionCopy(current_x, e);

            // assemble residual
            QuadLoop(current_element_x,
                     i_thread,
                     e.quad_data_,
                     tmp,
                     local_residual);

            double* grad_data = local_grad.GetData();
            double* solution_data = current_element_x.GetData();
            const double* residual_data = local_residual.GetData();
            const double* fd_forward_data = tmp.forward_residual_.GetData();
            for (int j{}; j < e.n_tdof_; ++j) {
              tmp.forward_residual_ = 0.0;

              double& with_respect_to = *solution_data++;
              const double orig_wrt = with_respect_to;
              const double diff_step = std::abs(orig_wrt) * 1.0e-8;
              const double diff_step_inv = 1. / diff_step;

              with_respect_to = orig_wrt + diff_step;
              QuadLoop(current_element_x,
                       i_thread,
                       e.quad_data_,
                       tmp,
                       tmp.forward_residual_);

              for (int k{}; k < e.n_tdof_; ++k) {
                *grad_data++ =
                    (fd_forward_data[k] - residual_data[k]) * diff_step_inv;
              }
              with_respect_to = orig_wrt;
            }

            // push right away
            std::lock_guard<std::mutex> lock(residual_mutex);
            const auto& vdofs = *e.v_dofs_;
            residual.AddElementVector(vdofs, local_residual.GetData());
            double* A = grad.GetData();
            const double* local_A = local_grad.GetData();
            const auto& A_ids = *precomputed_->domain_A_ids_[i];
            for (int k{}; k < A_ids.size(); ++k) {
              A[A_ids[k]] += *local_A++ * grad_factor;
            }
          }
        };

    mimi::utils::NThreadExe(assemble_element_residual_and_grad_then_contribute,
                            n_elements_,
                            (nthreads < 0) ? n_threads_ : nthreads);
  };

  /// Create mass matrix if we don't have one.
  virtual void CreateMassMatrix(const int size) {

    // size check
    if (m_mat_) {
      // if size differs (it shouldn't), remove
      if (m_mat_->Height() != size) {
        m_mat_ = nullptr;
      }
    }

    // prepare mass matrix
    if (!m_mat_) {
      m_mat_ = std::make_unique<mfem::SparseMatrix>(size);

      mfem::DenseMatrix mat;

      for (auto& e : element_data_) {
        mat.SetSize(e.n_dof_);
        mat = 0.0;

        for (auto& q_data : e.quad_data_) {
          mfem::AddMult_a_VVt(q_data.integration_weight_ * q_data.det_dX_dxi_,
                              q_data.N_,
                              mat);
        }
        m_mat_->AddSubMatrix(e.scalar_v_dofs_, e.scalar_v_dofs_, mat, 0);
      }

      m_mat_->Finalize();
      m_mat_->SortColumnIndices();
      // mass_inv_.iterative_mode = fal se;
      // mass_inv_.SetRelTol(1e-12);
      // // mass_inv_.SetAbsTol(1e-12);
      // mass_inv_.SetMaxIter(5000);
      // mass_inv_.SetPrintLevel(mfem::IterativeSolver::PrintLevel().All());
      // mass_inv_.SetPreconditioner(mass_inv_prec_);
      // mass_inv_.SetOperator(*m_mat_);

      mass_inv_direct_.SetOperator(*m_mat_);
      mass_inv_direct_.SetPrintLevel(1);
    }
  }

  /// pure integration of temperature at quadrature points
  virtual void Temperature(mfem::Vector& x, mfem::Vector& projected) {
    MIMI_FUNC()

    CreateMassMatrix(projected.Size());

    integrated_.SetSize(projected.Size());

    auto post_process =
        [&](const int begin, const int end, const int i_thread) {
          for (int i{begin}; i < end; ++i) {
            // in
            ElementData& e = element_data_[i];
            e.scalar_post_process_view_ = 0.0;

            for (auto& q_data : e.quad_data_) {
              e.scalar_post_process_view_.Add(
                  q_data.integration_weight_ * q_data.det_dX_dxi_
                      * q_data.material_state_
                            ->scalars_[J2AdiabaticVisco::State::k_temperature],
                  q_data.N_);
            }
          }
        };

    mimi::utils::NThreadExe(post_process, n_elements_, n_threads_);

    // serial assemble to integrated
    integrated_ = 0.0;
    for (auto& e_data : element_data_) {
      integrated_.AddElementVector(e_data.scalar_v_dofs_,
                                   e_data.scalar_post_process_view_);
    }

    // mass_inv_.Mult(integrated, projected);
    projected = 0.0;
    mass_inv_direct_.Mult(integrated_, projected);
  }

  virtual void AccumulatedPlasticStrain(mfem::Vector& x,
                                        mfem::Vector& projected) {
    MIMI_FUNC()

    CreateMassMatrix(projected.Size());

    integrated_.SetSize(projected.Size());

    auto post_process =
        [&](const int begin, const int end, const int i_thread) {
          for (int i{begin}; i < end; ++i) {
            // in
            ElementData& e = element_data_[i];
            e.scalar_post_process_view_ = 0.0;

            for (auto& q_data : e.quad_data_) {
              e.scalar_post_process_view_.Add(
                  q_data.integration_weight_ * q_data.det_dX_dxi_
                      * q_data.material_state_
                            ->scalars_[J2NonlinearIsotropicHardening::State::
                                           k_plastic_strain],
                  q_data.N_);
            }
          }
        };

    mimi::utils::NThreadExe(post_process, n_elements_, n_threads_);

    // serial assemble to integrated
    integrated_ = 0.0;
    for (auto& e_data : element_data_) {
      integrated_.AddElementVector(e_data.scalar_v_dofs_,
                                   e_data.scalar_post_process_view_);
    }

    // mass_inv_.Mult(integrated, projected);
    projected = 0.0;
    mass_inv_direct_.Mult(integrated_, projected);
  }
};

} // namespace mimi::integrators
