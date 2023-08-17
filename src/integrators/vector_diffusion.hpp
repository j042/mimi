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
  const mfem::Coefficient& coeff_;
  std::shared_ptr<FiniteElementToId_> fe_to_id_;

public:
  using ElementMatrices_ = mimi::utils::Data<mfem::DenseMatrix>;
  using FiniteElementToId_ = std::unordered_map<mfem::FiniteElement*, int>;

  /// nthread element holder
  std::unique_ptr<ElementMatrices_> element_matrices_;

  /// map from finite element to its id
  std::shared_ptr<FiniteElementToId_> fe_to_id_;

  /// ctor
  VectorMass(const mfem::Coefficient& coeff,
             const std::shared_ptr<FiniteElementToId_>& fe_to_id_)
      : coeff_(coeff),
        fe_to_id_(fe_to_id_) {}

  /// @brief Precomputes matrix.
  /// @param fes
  /// @param nthreads
  void ComputeElementMatrices(const mfem::FiniteElementSpace& fes,
                              const int nthreads) {
    MIMI_FUNC();
    // this integrator does not support embedded problems. mfem's does
    assert(fes->GetFE(0)->GetDim()
           == fes->GetElementTransformation(0)->GetSpaceDim());

    const int n_elem = fes.GetNE();

    // allocate
    element_matrices_ = std::make_unique<ElementMatrices_>(n_elem);

    auto assemble_element_matrices =
        [&](const int begin, const int end, const int) {
          // aux mfem containers
          mfem::DenseMatrix d_shape, d_shape_dxt, p_elmat;

          for (int i{begin}; i < end; ++i) {
            // get related objects from fespace
            const mfem::FiniteElement& el = *fes->GetFE(i);
            mfem::ElementTransformation& eltrans_stress_free_to_reference =
                *fes.GetElementTransformation(i);

            // basic infos for this elem
            const int n_dof = el.GetDof();
            const int dim = el.GetDim();

            // alloc shape and shape_mat
            d_shape.SetSize(n_dof, dim);
            d_shape_dxt.SetSize(n_dof, n_dof);
            p_elmat.SetSize(n_dof, n_dof);

            // get elmat to save and set size
            mfem::DenseMatrix& elmat = element_matrices_[i];
            elmat.SetSize(n_dof * dim, n_dof * dim);
            elmat = 0.0;

            // prepare quad loop
            const mfem::IntegrationRule& ir = mfem::IntRules.Get(
                el.GetGeomType(),
                2 * el.GetOrder() + eltrans_stress_free_to_reference.OrderW());

            // quad loop
            for (int q{}; q < ir.GetNPoints(); ++q) {
              const mfem::IntegrationPoint& ip = ir.IntPoint(q);

              // get d_shape
              el.CalcDShape(ip, d_shape);

              // get d_shape_dxt
              mfem::Mult(d_shape,
                         eltrans_stress_free_to_reference.AdjugateJacobian(),
                         d_shape_ext);

              // prepare transformation
              eltrans_stress_free_to_reference.SetIntPoint(&ip);

              // save a weight for integration
              const double weight {
                ip.weight / eltrans_stress_free_to_reference.Weight()
              }

              // aux variable factor is weight times coeff
              // again, this only supports scalar coeff. for vector/matrix
              // extend here.
              const double factor =
                  coeff_->Eval(eltrans_stress_free_to_reference, ip) * weight;

              // compute contribution
              mfem::Mult_a_AAt(factor, d_shape_dxt, p_elmat);

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
                                     ElementTransformation& eltrans,
                                     DenseMatrix& elmat) {
    // return saved values
    auto& saved = element_matrices_->operator[](fe_to_id_->operator[](&el));
    elmat.UseExternalData(saved.GetData(), saved.Height(), saved.Width());
  }
}

} // namespace mimi::integrators
