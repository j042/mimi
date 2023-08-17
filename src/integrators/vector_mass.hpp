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

  /// @brief Precomputes matrix. After writing this, notices MFEM, of course
  /// has a similar option using OpenMP.
  /// @param fes
  /// @param nthreads
  void ComputeElementMatrices(const mfem::FiniteElementSpace& fes,
                              const int nthreads const bool invert = false) {
    MIMI_FUNC();

    const int n_elem = fes.GetNE();

    // allocate
    element_matrices_ = std::make_unique<ElementMatrices_>(n_elem);

    auto assemble_element_matrices =
        [&](const int begin, const int end, const int) {
          // aux mfem containers
          mfem::Vector shape;
          mfem::DenseMatrix shape_mat;

          for (int i{begin}; i < end; ++i) {
            // get related objects from fespace
            const mfem::FiniteElement& el = *fes->GetFE(i);
            mfem::ElementTransformation& eltrans_stress_free_to_reference =
                *fes.GetElementTransformation(i);

            // basic infos for this elem
            const int n_dof = el.GetDof();
            const int dim = el.GetDim();

            // alloc shape and shape_mat
            shape.SetSize(n_dof);
            shape_mat.SetSize(n_dof, n_dof);

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

              // get shape
              el.CalcShape(ip, shape);

              // prepare transformation
              eltrans_stress_free_to_reference.SetIntPoint(&ip);

              // save a weight for integration
              const double weight{ip.weight
                                  * eltrans_stress_free_to_reference.Weight()}

              // VVt
              mfem::MultVVt(shape, shape_mat);

              // here, we only have ctor for coeff. but if Vector/Matrix coeff
              // is planned implement changes here
              shape_mat *=
                  weight * coeff_->Eval(eltrans_stress_free_to_reference, ip);

              // four for loops
              for (int d{}; d < dim; ++d) {
                elmat.AddMatrix(shape_mat, n_dof * d, n_dof * d);
              }
            } // quad loop

            if (invert) {
              elmat.Invert();
            }
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
