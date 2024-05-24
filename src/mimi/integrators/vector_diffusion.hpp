#pragma once

#include <memory>
#include <unordered_map>
#include <utility>

#include <mfem.hpp>
// #include <mfem/fem/nonlininteg.hpp>

#include "mimi/utils/containers.hpp"
#include "mimi/utils/n_thread_exe.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

/// Rewrite of mfem::VectorDifussionIntegrator
/// (Q grad u, grad v) = sum_i (Q grad u_i, grad v_i) e_i e_i^T
class VectorDiffusion : public mfem::BilinearFormIntegrator {
protected:
  std::shared_ptr<mfem::Coefficient> coeff_;

public:
  using ElementMatrices_ = mimi::utils::Data<mfem::DenseMatrix>;
  using FiniteElementToId_ = std::unordered_map<mfem::FiniteElement*, int>;

  /// nthread element holder
  std::unique_ptr<ElementMatrices_> element_matrices_;

  /// ctor
  VectorDiffusion(const std::shared_ptr<mfem::Coefficient>& coeff)
      : coeff_(coeff) {}

  /// @brief Precomputes matrix.
  /// @param fes
  /// @param nthreads
  void ComputeElementMatrices(mimi::utils::PrecomputedData& precomputed) {
    MIMI_FUNC()

    auto& fes = *precomputed.fe_spaces_[0];

    // this integrator does not support embedded problems. mfem's does
    assert(fes.GetFE(0)->GetDim()
           == fes.GetElementTransformation(0)->GetSpaceDim());

    const int n_elem = fes.GetNE();
    const int nthreads = precomputed.fe_spaces_.size();

    // allocate
    element_matrices_ = std::make_unique<ElementMatrices_>(n_elem);

    auto assemble_element_matrices = [&](const int begin,
                                         const int end,
                                         const int i_thread) {
      // aux mfem containers
      mfem::DenseMatrix d_shape, d_shape_dxt, p_elmat;

      for (int i{begin}; i < end; ++i) {
        auto& int_rules = precomputed.int_rules_flat_[i_thread];
        auto& el = *precomputed.elements_[i];
        auto& eltrans_reference_to_target =
            *precomputed.reference_to_target_element_trans_flat_[i];

        // basic infos for this elem
        const int n_dof = el.GetDof();
        const int dim = el.GetDim();

        // alloc shape and shape_mat
        d_shape.SetSize(n_dof, dim);
        d_shape_dxt.SetSize(n_dof, dim);
        p_elmat.SetSize(n_dof, n_dof);

        // get elmat to save and set size
        mfem::DenseMatrix& elmat = element_matrices_->operator[](i);
        elmat.SetSize(n_dof * dim, n_dof * dim);
        elmat = 0.0;

        // prepare quad loop
        const mfem::IntegrationRule& ir = mfem::IntRules.Get(
            el.GetGeomType(),
            2 * el.GetOrder() + eltrans_reference_to_target.OrderW());

        // quad loop
        for (int q{}; q < ir.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(q);

          // get d_shape
          el.CalcDShape(ip, d_shape);

          // get d_shape_dxt
          mfem::Mult(d_shape,
                     eltrans_reference_to_target.AdjugateJacobian(),
                     d_shape_dxt);

          // prepare transformation
          eltrans_reference_to_target.SetIntPoint(&ip);

          // save a weight for integration
          const double weight{ip.weight / eltrans_reference_to_target.Weight()};

          // aux variable factor is weight times coeff
          // again, this only supports scalar coeff. for vector/matrix
          // extend here.
          const double factor =
              coeff_->Eval(eltrans_reference_to_target, ip) * weight;

          // compute contribution
          mfem::Mult_a_AAt(factor, d_shape_dxt, p_elmat);

          // four for loops
          for (int d{}; d < dim; ++d) {
            elmat.AddMatrix(p_elmat, n_dof * d, n_dof * d);
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
