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

/// Rewrite of mfem::VectorMassIntegrator
/// a(u, v) = (Q u, v)
class VectorMass : public mfem::BilinearFormIntegrator {
protected:
  std::shared_ptr<mfem::Coefficient> coeff_;

public:
  using ElementMatrices_ = mimi::utils::Data<mfem::DenseMatrix>;
  using FiniteElementToId_ = std::unordered_map<mfem::FiniteElement*, int>;

  /// nthread element holder
  std::unique_ptr<ElementMatrices_> element_matrices_;

  /// ctor
  VectorMass(const std::shared_ptr<mfem::Coefficient>& coeff) : coeff_(coeff) {}

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
      mfem::Vector shape;
      mfem::DenseMatrix shape_mat;

      for (int i{begin}; i < end; ++i) {
        auto& int_rules = precomputed.int_rules_[i_thread];
        auto& el = *precomputed.elements_flat_[i];
        auto& eltrans_reference_to_target =
            *precomputed.reference_to_target_element_trans_flat_[i];

        // basic infos for this elem
        const int n_dof = el.GetDof();
        const int dim = el.GetDim();

        // alloc shape and shape_mat
        shape.SetSize(n_dof);
        shape_mat.SetSize(n_dof, n_dof);

        // get elmat to save and set size
        mfem::DenseMatrix& elmat = element_matrices_->operator[](i);
        elmat.SetSize(n_dof * dim, n_dof * dim);
        elmat = 0.0;

        // prepare quad loop
        const mfem::IntegrationRule& ir = int_rules->Get(
            el.GetGeomType(),
            2 * el.GetOrder() + eltrans_reference_to_target.OrderW());

        // quad loop
        for (int q{}; q < ir.GetNPoints(); ++q) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(q);

          // get shape
          el.CalcShape(ip, shape);

          // prepare transformation
          eltrans_reference_to_target.SetIntPoint(&ip);

          // save a weight for integration
          const double weight{ip.weight * eltrans_reference_to_target.Weight()};

          // VVt
          mfem::MultVVt(shape, shape_mat);

          // here, we only have ctor for coeff. but if Vector/Matrix coeff
          // is planned implement changes here
          shape_mat *= weight * coeff_->Eval(eltrans_reference_to_target, ip);

          // four for loops
          for (int d{}; d < dim; ++d) {
            elmat.AddMatrix(shape_mat, n_dof * d, n_dof * d);
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
