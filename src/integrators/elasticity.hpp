#pragma once

#include <utility>

#include <mfem.hpp>
// #include <mfem/fem/nonlininteg.hpp>

#include "mimi/utils/default_init_vector.hpp"

namespace mimi::integrators {

/// @brief Young's modulus and poisson's ratio to lambda and mu
/// @param young
/// @param poisson
/// @param lambda
/// @param mu
std::pair<double, double> ToLambdaAndMu(const double young,
                                        const double poisson) {
  return {young / (3.0 * (1.0 - (2.0 * poisson))), // lambda
          young / (2.0 * (1.0 + poisson))};        // mu
}

/// Rewrite of mfem::ElasticityIntegrator
/// a(u, v) = (lambda div(u), div(v)) + (2 * mu * e(u), e(v)),
/// where e(v) = (1/2) * (grad(v) * grad(v)^T)
class LinearElasticity : public mfem::NonlinearFormIntegrator {
protected:
  mfem::Coefficient& lambda_;
  mfem::Coefficient& mu_;

  // in case you don't need this to be thread safe, feel free to use the
  // following values
  mfem::Vector shape_, div_shape_;
  mfem::DenseMatrix d_shape_, g_shape_, p_elmat_;

  mfem::IntegrationRule ir;
  mimi::utils::Vector<mfem::DenseMatrix> saved_elmat_;

  /// Precompute Element Matrix
  void PrecomputeElementMatrix(const mfem::FiniteElementSpace& fes,
                               const int nthreads) {
    const int n_elem = fes.GetNE();

    // allocate
    saved_elmat_.resize(n_elem);

    auto assemble_element_matrix = [&](int begin, int end) {
      // aux mfem containers
      mfem::DenseMatrix d_shape, g_shape, p_elmat;
      mfem::Vector div_shape;
      for (int i{begin}; i < end; ++i) {
        // get related objects from fespace
        const mfem::FiniteElement& el = *fes->GetFE(i);
        mfem::ElementTransformation& eltrans_stress_free_to_reference =
            *fes.GetElementTransformation(i);

        // basic infos for this elem
        const int n_dof = el.GetDof();
        const int dim = el.GetDim();

        // set size for aux containers
        d_shape.SetSize(n_dof, dim);
        g_shape.SetSize(n_dof, dim);
        p_elmat.SetSize(n_dof, n_dof); // same as p_elmat.SetSize(n_dof);
        div_shape.SetSize(dim * dof);

        // get elmat to save and set size
        mfem::DenseMatrix& elmat = saved_elmat_[i];
        elmat.SetSize(n_dof * dim, n_dof * dim);
        elmat = 0.0;

        // prepare quad loop;
        const mfem::IntegrationRule& ir =
            mfem::IntRules.Get(el.GetGeomType(), 2 * el.GetOrder() + 3);

        // quad loop
        for (int i{}; i < ir.GetNPoints(); ++i) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(i);

          // get first der of basis functions from each support
          el.CalcDShape(ip, d_shape);

          // prepare transformation
          eltrans_stress_free_to_reference.SetIntPoint(&ip);

          // save a weight for integration
          const double weight{ip.weight
                              * eltrans_stress_free_to_reference.Weight()}

          // get geometric d shape
          mfem::Mult(d_shape,
                     eltrans_stress_free_to_reference.InverseJacobian(),
                     g_shape);

          // get p_elmat - what's this?
          mfem::MultAAt(g_shape, p_elmat)

              // get div - this is just flat view of g_shape, but useful for VVt
              g_shape.GradToDiv(div_shape);

          // prepare params
          const double lambda =
              lambda_.Eval(eltrans_stress_free_to_reference, ip);
          const double mu = mu_.Eval(eltrans_stress_free_to_reference, ip);

          // we assume lambda is none zero
          mfem::AddMult_a_VVt(lambda * w, div_shape, elmat);

          // mu also
          const double mu_times_weight = mu * weight;
          for (int d{}; d < dim; ++d) {
            for (int dof_i{}; dof_i < n_dof; ++dof_i) {
              for (int dof_j{}; dof_j < n_dof; ++dof_j) {
                const int offset = n_dof * d;
                elmat(offset + dof_i, offset + dof_j) +=
                    mu_times_weight * p_elmat(dof_i, dof_j);
              }
            }
          }

          for (int dim_i{}; dim_i < dim; ++dim_i) {
            for (int dim_j{}; dim_j < dim; ++dim_j) {
              for (int dof_i{}; dof_i < n_dof; ++dof_i) {
                for (int dof_j{}; dof_j < n_dof; ++dof_j) {
                  elmat(n_dof * dim_i + dof_i, n_dof * dim_j + dof_j) +=
                      mu_times_weight * g_shape(dof_i, dim_j)
                      * gshape(dof_j, dim_i);
                }
              }
            }
          }

        } // quad loop
      }
    };
  }

  /// @brief Assembles and saves.
  /// @param el
  /// @param eltrans
  /// @param elmat
  void AssembleElementMatrix(const mfem::FiniteElement& el,
                             ElementTransformation& eltrans,
                             DenseMatrix& elmat) {}
}

} // namespace mimi::integrators
