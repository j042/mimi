#pragma once

#include <memory>
#include <unordered_map>
#include <utility>

#include <mfem.hpp>
// #include <mfem/fem/nonlininteg.hpp>

#include "mimi/utils/containers.hpp"
#include "mimi/utils/n_thread_exe.hpp"
#include "mimi/utils/precomputed.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

/// Rewrite of mfem::ElasticityIntegrator
/// a(u, v) = (lambda div(u), div(v)) + (2 * mu * e(u), e(v)),
/// where e(v) = (1/2) * (grad(v) * grad(v)^T)
class LinearElasticity : public mfem::BilinearFormIntegrator {
protected:
  std::shared_ptr<mfem::Coefficient> lambda_;
  std::shared_ptr<mfem::Coefficient> mu_;

public:
  using ElementMatrices_ = mimi::utils::Data<mfem::DenseMatrix>;
  using FiniteElementToId_ = std::unordered_map<mfem::FiniteElement*, int>;

  /// nthread element holder
  std::unique_ptr<ElementMatrices_> element_matrices_;

  /// ctor
  LinearElasticity(const std::shared_ptr<mfem::Coefficient>& lambda,
                   const std::shared_ptr<mfem::Coefficient>& mu)
      : lambda_(lambda),
        mu_(mu) {}

  /// @brief Precomputes matrix. After writing this, notices MFEM, of course
  /// has a similar option using OpenMP.
  /// @param fes
  /// @param nthreads
  void ComputeElementMatrices(mimi::utils::PrecomputedData& precomputed) {
    MIMI_FUNC()

    const int n_elem = precomputed.fe_spaces_[0]->GetNE();
    const int nthreads = precomputed.fe_spaces_.size();
    // allocate
    element_matrices_ = std::make_unique<ElementMatrices_>(n_elem);

    auto assemble_element_matrices = [&](const int begin,
                                         const int end,
                                         const int i_thread) {
      // aux mfem containers
      mfem::DenseMatrix d_shape, g_shape, p_elmat;
      mfem::Vector div_shape;

      for (int i{begin}; i < end; ++i) {
        auto& int_rules = precomputed.int_rules_[i_thread];
        auto& el = *precomputed.elements_[i];
        auto& eltrans_reference_to_target =
            *precomputed.reference_to_target_element_trans_[i];

        // basic infos for this elem
        const int n_dof = el.GetDof();
        const int dim = el.GetDim();

        // set size for aux containers
        d_shape.SetSize(n_dof, dim);
        g_shape.SetSize(n_dof, dim);
        p_elmat.SetSize(n_dof, n_dof); // same as p_elmat.SetSize(n_dof);
        div_shape.SetSize(dim * n_dof);

        // get elmat to save and set size
        mfem::DenseMatrix& elmat = element_matrices_->operator[](i);
        elmat.SetSize(n_dof * dim, n_dof * dim);
        elmat = 0.0;

        // prepare quad loop;
        const mfem::IntegrationRule& ir =
            int_rules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3);

        // quad loop
        for (int q{}; q < ir.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(q);

          // get first der of basis functions from each support
          el.CalcDShape(ip, d_shape);

          // prepare transformation
          eltrans_reference_to_target.SetIntPoint(&ip);

          // save a weight for integration
          const double weight{ip.weight * eltrans_reference_to_target.Weight()};

          // get geometric d shape
          mfem::Mult(d_shape,
                     eltrans_reference_to_target.InverseJacobian(),
                     g_shape);

          // get p_elmat - nabla u * nabla vT
          mfem::MultAAt(g_shape, p_elmat);

          // get div - this is just flat view of g_shape, but useful for VVt
          g_shape.GradToDiv(div_shape);

          // prepare params
          const double lambda = lambda_->Eval(eltrans_reference_to_target, ip);
          const double mu = mu_->Eval(eltrans_reference_to_target, ip);

          // add lambda
          mfem::AddMult_a_VVt(lambda * weight, div_shape, elmat);
          const double mu_times_weight = mu * weight;

          for (int d{}; d < dim; ++d) {
            const int offset = n_dof * d;
            for (int dof_i{}; dof_i < n_dof; ++dof_i) {
              for (int dof_j{}; dof_j < n_dof; ++dof_j) {
                elmat(offset + dof_i, offset + dof_j) +=
                    mu_times_weight * p_elmat(dof_i, dof_j);
              }
            }
          }

          for (int dim_i{}; dim_i < dim; ++dim_i) {
            const int offset_i = n_dof * dim_i;
            for (int dim_j{}; dim_j < dim; ++dim_j) {
              const int offset_j = n_dof * dim_j;
              for (int dof_i{}; dof_i < n_dof; ++dof_i) {
                for (int dof_j{}; dof_j < n_dof; ++dof_j) {
                  elmat(offset_i + dof_i, offset_j + dof_j) +=
                      mu_times_weight * g_shape(dof_i, dim_j)
                      * g_shape(dof_j, dim_i);
                }
              }
            }
          }
        } // quad loop
      }
    };

    // exe
    mimi::utils::NThreadExe(assemble_element_matrices, n_elem, nthreads);
  }

  /// @brief returns copy of assembled matrices
  /// @param el
  /// @param eltrans
  /// @param elmat
  virtual void AssembleElementMatrix(const mfem::FiniteElement& el,
                                     mfem::ElementTransformation& eltrans,
                                     mfem::DenseMatrix& elmat) {
    MIMI_FUNC()

    // return saved values
    auto& saved = element_matrices_->operator[](eltrans.ElementNo);
    elmat.UseExternalData(saved.GetData(), saved.Height(), saved.Width());
  }
};

} // namespace mimi::integrators
