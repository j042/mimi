#pragma once

#include <cmath>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/coefficients/nearest_distance.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

/// normal - we define a rule here, and it'd be your job to prepare
/// the foreign spline in a correct orientation
template<int dim, bool unit_normal = true, typename ArrayType>
inline void Normal(const mimi::utils::Data<double, 2>& first_dir,
                   ArrayType& normal) {
  assert(first_dir.size() == 2 || first_dir.size() == 6);

  if constexpr (dim == 2) {
    const double& d0 = first_dir[0];
    const double& d1 = first_dir[1];

    if constexpr (unit_normal) {
      const double inv_norm2 = 1. / std::sqrt(d0 * d0 + d1 * d1);

      normal[0] = d1 * inv_norm2;
      normal[1] = -d0 * inv_norm2;
    } else {
      normal[0] = d1;
      normal[1] = -d0;
    }

  } else if constexpr (dim == 3) {
    const double& d0 = first_dir[0];
    const double& d1 = first_dir[1];
    const double& d2 = first_dir[2];
    const double& d3 = first_dir[3];
    const double& d4 = first_dir[4];
    const double& d5 = first_dir[5];

    if constexpr (unit_normal) {
      const double n0 = d1 * d5 - d2 * d4;
      const double n1 = d2 * d3 - d0 * d5;
      const double n2 = d0 * d4 - d1 * d3;

      const double inv_norm2 = 1. / std::sqrt(n0 * n0 + n1 * n1 + n2 * n2);

      normal[0] = n0 * inv_norm2;
      normal[1] = n1 * inv_norm2;
      normal[2] = n2 * inv_norm2;

    } else {

      normal[0] = d1 * d5 - d2 * d4;
      normal[1] = d2 * d3 - d0 * d5;
      normal[2] = d0 * d4 - d1 * d3;
    }
  } else {
    static_assert(dim == 2 || dim == 3, "unsupported dim");
  }
}

template<typename ArrayType>
inline double
NormalGap(const mimi::coefficients::NearestDistanceBase::Results& result,
          ArrayType& normal) {

  double normal_gap{};
  const int& dim = result.dim_;

  // let's get normal
  if (dim == 2) {
    Normal<2, true>(result.first_derivatives_, normal);
  } else {
    Normal<3, true>(result.first_derivatives_, normal);
  }

  for (int i{}; i < dim; ++i) {
    // here, we apply negative sign to physical_minus_query
    // normal gap is formulated as query minus physical
    normal_gap += normal[i] * -result.physical_minus_query_[i];
  }

  return normal_gap;
}

class PenaltyContact : public NonlinearBase {
protected:
  /// scene
  std::shared_ptr<mimi::coefficients::NearestDistanceBase>
      nearest_distance_coeff_ = nullptr;

  /// @brief quadrature order per elements - alternatively, could be per
  /// boundary patch thing
  mimi::utils::Vector<int> quadrature_orders_;

  /// results from proximity queries, to be reused for grad - latest relevant
  /// residual and grad are calls right after one another from newton solver
  mimi::utils::Vector<
      mimi::utils::Vector<mimi::coefficients::NearestDistanceBase::Results>>
      nearest_distance_results_;

  /// decided to ask for intrules all the time. but since we don't wanna call
  /// the geometry type all the time, we save this just once.
  mfem::Geometry::Type boundary_geometry_type_;

  /// convenient constants - space dim
  int dim_;
  int boundary_para_dim_;

public:
  using Base_ = NonlinearBase;

  PenaltyContact(
      const std::shared_ptr<mimi::coefficients::NearestDistanceBase>&
          nearest_distance_coeff,
      const std::string& name,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : NonlinearBase(name, precomputed),
        nearest_distance_coeff_(nearest_distance_coeff) {}

  /// this one needs
  /// - shape function
  /// - weights of target to refrence
  virtual void Prepare(const int quadrature_order = -1) {
    MIMI_FUNC()

    // get numbers to decide loop size
    const int n_boundary_elements =
        precomputed_->meshes_[0]->NURBSext->GetNBE();
    const int n_threads =
        precomputed_->meshes_.size(); // size of mesh is nthread from Setup()

    // get dimension
    dim_ = precomputed_->meshes_[0]->Dimension();
    boundary_para_dim_ = dim_ - 1;

    // get marked boundary elements: this reduces size of the loop
    if (!boundary_marker_) {
      mimi::utils::PrintAndThrowError(Name(),
                                      "does not have a boundary marker.");
    }
    // prepare loop - deref and get a mesh to ask
    auto& mesh = *precomputed_->meshes_[0];
    auto& b_marker = *boundary_marker_;
    Base_::marked_boundary_elements_.reserve(n_boundary_elements);

    // loop boundary elements
    for (int i{}; i < n_boundary_elements; ++i) {
      auto& b_el = precomputed_->boundary_elements_[i];
      const int bdr_attr = mesh.GetBdrAttribute(i);
      if (b_marker[bdr_attr - 1] == 0) {
        continue;
      }
      Base_::marked_boundary_elements_.push_back(i);
    }
    Base_::marked_boundary_elements_.shrink_to_fit();

    // extract boundary geometry type
    boundary_geometry_type_ =
        precomputed_->boundary_elements_[0]->GetGeomType();

    // allocate result holder
    nearest_distance_results_.resize(n_boundary_elements);

    // allocate element vectors and matrices
    Base_::boundary_element_matrices_ =
        std::make_unique<mimi::utils::Data<mfem::DenseMatrix>>();
    Base_::boundary_element_matrices_->SetSize(n_boundary_elements);
    Base_::boundary_element_vectors_ =
        std::make_unique<mimi::utils::Data<mfem::Vector>>();
    Base_::boundary_element_vectors_->SetSize(n_boundary_elements);

    // shapes
    auto& boundary_shapes = precomputed_->vectors_["boundary_shapes"];
    boundary_shapes.resize(n_boundary_elements);

    // Dshapes - need to evaluate derivative at current quad point
    auto& boundary_d_shapes = precomputed_->matrices_["boundary_d_shapes"];
    boundary_d_shapes.resize(n_boundary_elements);

    // jacobian weights
    auto& boundary_target_to_reference_weights =
        precomputed_->scalars_["boundary_target_to_reference_weights"];
    boundary_target_to_reference_weights.resize(n_boundary_elements);

    // lagrange_multiplier for augmented
    // we don't have to use it, but we have it
    auto& augmented_lagrange_multipliers =
        precomputed_->scalars_["augmented_lagrange_multipliers"];
    augmented_lagrange_multipliers.resize(n_boundary_elements);

    // save normal gap to update lagrange multiplier
    // we actually need normal_distance which is just -normal_gap.
    // I just don't want to introduce a new variable.
    auto& normal_gaps = precomputed_->scalars_["normal_gaps"];
    normal_gaps.resize(n_boundary_elements);

    // quad order
    quadrature_orders_.resize(n_boundary_elements);

    // // shapes is once per boundary patch. yes, this will have many recurring
    // // shapes, which makes it easy to lookup
    // auto precompute_shapes = [&](const int b_patch_begin,
    //                              const int b_patch_end,
    //                              const int i_thread) {
    //   // thread's obj
    //   auto& int_rules = precomputed_->int_rules_[i_thread];

    //   for (int i{b_patch_begin}; i < b_patch_end; ++i) {
    //     // get corresponding vector of shapes
    //     auto& shapes = boundary_shapes[i];
    //     auto& d_shapes = boundary_d_shapes[i];

    //     // get id of a patch
    //     mimi::utils::PrintInfo(precomputed_->meshes_[0]->NURBSext->GetNP());
    //     const int b_el_id =
    //         precomputed_->meshes_[0]->NURBSext->GetPatchBdrElements(i)[0];

    //     // get elem
    //     auto& b_el = *precomputed_->boundary_elements_[b_el_id];
    //     const int n_dof = b_el.GetDof();

    //     // get quad order
    //     const int q_order =
    //         (quadrature_order < 0) ? b_el.GetOrder() * 2 + 3 :
    //         quadrature_order;

    //     // get int rule
    //     const mfem::IntegrationRule& ir =
    //         int_rules.Get(boundary_geometry_type_, q_order);
    //     // prepare quad loop
    //     const int n_quad = ir.GetNPoints();
    //     shapes.resize(n_quad);

    //     // quad loop - calc shapes
    //     for (int j{}; j < n_quad; ++j) {
    //       mfem::Vector& shape = shapes[j];
    //       mfem::DenseMatrix& d_shape = d_shapes[j];
    //       shape.SetSize(n_dof);
    //       d_shape.SetSize(n_dof, boundary_para_dim_);

    //       const mfem::IntegrationPoint& ip = ir.IntPoint(j);
    //       b_el.CalcShape(ip, shape);
    //       b_el.CalcDShape(ip, d_shape);
    //     }
    //   }
    // };

    // now, weight of jacobian.
    auto precompute_trans_weights = [&](const int marked_b_el_begin,
                                        const int marked_b_el_end,
                                        const int i_thread) {
      // thread's obj
      auto& int_rules = precomputed_->int_rules_[i_thread];

      for (int i{marked_b_el_begin}; i < marked_b_el_end; ++i) {
        // we only need makred boundaries
        const int& i_mbe = Base_::marked_boundary_elements_[i];

        // get element
        const auto& i_b_el = precomputed_->boundary_elements_[i_mbe];

        // shapes, dshapes
        auto& i_shapes = boundary_shapes[i_mbe];
        auto& i_d_shapes = boundary_d_shapes[i_mbe];

        // get boundary transformations
        auto& i_b_trans =
            *precomputed_->target_to_reference_boundary_trans_[i_mbe];

        // get corresponding vector of weights to fill
        auto& i_weights = boundary_target_to_reference_weights[i_mbe];

        // get corresponding vector of lagrange multipliers
        // TODO: check if per-quad is correct.
        auto& i_lagrange = augmented_lagrange_multipliers[i_mbe];

        // get corresponding vector of normal_gaps
        auto& i_normal_gaps = normal_gaps[i_mbe];

        // get results holder to allocate
        auto& i_results = nearest_distance_results_[i_mbe];

        // get quad order
        const int q_order = (quadrature_order < 0)
                                ? i_b_trans.GetFE()->GetOrder() * 2 + 3
                                : quadrature_order;

        // save this quad_order for the element
        quadrature_orders_[i_mbe] = q_order;

        // get int rule
        const mfem::IntegrationRule& ir =
            int_rules.Get(boundary_geometry_type_, q_order);
        // prepare quad loop
        const int n_quad = ir.GetNPoints();

        // ndof
        const int n_dof = i_b_el->GetDof();

        // allocate boundary element based values
        Base_::boundary_element_vectors_->operator[](i_mbe).SetSize(n_dof
                                                                    * dim_);
        Base_::boundary_element_matrices_->operator[](i_mbe).SetSize(
            n_dof * dim_,
            n_dof * dim_);

        // use n_quad to allocate quad dependant values
        i_weights.resize(n_quad);
        i_results.resize(n_quad);
        i_lagrange.resize(n_quad, 0.0);
        i_normal_gaps.resize(n_quad, 0.0);
        i_shapes.resize(n_quad);
        i_d_shapes.resize(n_quad);

        // loop quad points
        for (int j{}; j < n_quad; ++j) {
          // start by setting int point
          const mfem::IntegrationPoint& ip = ir.IntPoint(j);
          i_b_trans.SetIntPoint(&ip);

          // get shapes and dshapes
          mfem::Vector& j_shape = i_shapes[j];
          mfem::DenseMatrix& j_d_shape = i_d_shapes[j];
          j_shape.SetSize(n_dof);
          j_d_shape.SetSize(n_dof, boundary_para_dim_);
          i_b_el->CalcShape(ip, j_shape);
          i_b_el->CalcDShape(ip, j_d_shape);

          // get trans weight
          i_weights[j] = i_b_trans.Weight();

          // alloate each members of results
          i_results[j].SetSize(boundary_para_dim_, dim_);
        }
      }
    };

    // mimi::utils::NThreadExe(precompute_shapes, n_boundary_patches,
    // n_threads);
    mimi::utils::NThreadExe(
        precompute_trans_weights,
        static_cast<int>(Base_::marked_boundary_elements_.size()),
        n_threads);
  }

  virtual void UpdateLagrange() {
    MIMI_FUNC();
    mimi::utils::PrintAndThrowError("noch nicht bereit");
  }

  virtual void AssembleBoundaryResidual(const mfem::Vector& current_x) {
    MIMI_FUNC()

    // get related precomputed values
    auto& boundary_shapes =
        precomputed_->vectors_["boundary_shapes"]; // n_patches * n_quad (n_dof)
                                                   // - each can vary in size
    auto& boundary_d_shapes =
        precomputed_->matrices_["boundary_d_shapes"]; // n_patches * n_quad
                                                      // (n_dof * n_dim)

    // jacobian weights
    auto& boundary_target_to_reference_weights =
        precomputed_
            ->scalars_["boundary_target_to_reference_weights"]; // n_b_elem *
                                                                // n_quad

    // augmented larange
    auto& augmented_lagrange_multipliers =
        precomputed_->scalars_["augmented_lagrange_multipliers"];

    // normal gaps
    auto& normal_gaps = precomputed_->scalars_["normal_gaps"];

    auto assemble_face_residual = [&](const int begin,
                                      const int end,
                                      const int i_thread) {
      // prepare per thread data - query, int_rule
      mimi::coefficients::NearestDistanceBase::Query query{};
      query.max_iterations_ = 20;
      query.query_.SetSize(dim_); // allocate

      auto& int_rules = precomputed_->int_rules_[i_thread];

      // temp arrays
      mimi::utils::Data<double> normal(dim_);
      mimi::utils::Data<double> traction_n(dim_);

      // element dependent densemat
      // this is always the same for each boundary of a single patch
      // we use densemat, becaue this has Weight() function implemented
      // which corresponds to sqrt(det(derivatives))
      mfem::DenseMatrix current_quad_derivatives(dim_, boundary_para_dim_);

      // convenient view of residual as matrix
      mfem::DenseMatrix i_residual_view;

      // this loops marked boundary elements
      for (int i{begin}; i < end; ++i) {
        // get this loop's objects
        // important to extract makred value.

        const int& i_mbe = Base_::marked_boundary_elements_[i];

        const auto& i_b_vdof = precomputed_->boundary_v_dofs_[i_mbe];
        auto& i_b_el = precomputed_->boundary_elements_[i_mbe];
        const auto& i_shapes = boundary_shapes[i_mbe];
        const auto& i_d_shapes = boundary_d_shapes[i_mbe];
        const auto& i_target_to_reference_weights =
            boundary_target_to_reference_weights[i_mbe];
        auto& i_results = nearest_distance_results_[i_mbe];
        auto& i_lagranges = augmented_lagrange_multipliers[i_mbe];
        auto& i_normal_gaps = normal_gaps[i_mbe];
        auto& i_residual = Base_::boundary_element_vectors_->operator[](i_mbe);
        i_residual = 0.0;

        // sizes
        const int n_dof = i_b_el->GetDof();

        // setup helper matrices
        current_quad_derivatives.SetSize(dim_, boundary_para_dim_);
        i_residual_view.UseExternalData(i_residual.GetData(), n_dof, dim_);

        assert(dim_ == i_results[0].dim_);
        assert(boundary_para_dim_ == i_b_el->GetDim());

        // copy current solution
        mfem::Vector i_current_solution_vec;
        current_x.GetSubVector(*i_b_vdof, i_current_solution_vec);
        mfem::DenseMatrix i_current_solution(i_current_solution_vec.GetData(),
                                             n_dof,
                                             dim_);

        // prepare quad loop
        const auto& int_rule =
            int_rules.Get(boundary_geometry_type_, quadrature_orders_[i_mbe]);

        // quad loop
        for (int q{}; q < int_rule.GetNPoints(); ++q) {

          const mfem::IntegrationPoint& ip = int_rule.IntPoint(q);
          const auto& q_shape = i_shapes[q];

          const auto& q_d_shape = i_d_shapes[q];

          const auto& q_target_to_reference_weight =
              i_target_to_reference_weights[q];

          const auto& q_lagrange = i_lagranges[q];

          auto& q_result = i_results[q];

          auto& q_normal_gap = i_normal_gaps[q];

          // mark this result inactive first so that we don't have to set
          // this at each early exit check
          q_result.active_ = false;

          // form query - get current position
          i_current_solution.MultTranspose(q_shape, query.query_.data());
          // evaluate derivatives at current quad position
          mfem::MultAtB(i_current_solution,
                        q_d_shape,
                        current_quad_derivatives);

          // query nearest distance
          nearest_distance_coeff_->NearestDistance(query, q_result);

          // get normal gap
          q_normal_gap = NormalGap(q_result, normal);

          // active checks
          //
          // 1. exact zero check. we put hold on this one
          // if (q_result.distance_ < nearest_distance_coeff_->tolerance_)

          // 2. normal gap orientation - we can probably merge condition (1)
          // here
          if (q_normal_gap > 0.) {
            mimi::utils::PrintDebug(
                "exiting quad loop because normal gap is positive");
            continue;
          }

          // 3. angle check
          constexpr const double angle_tolerance = 1.0e-5;
          if (std::acos(
                  std::min(1., std::abs(q_normal_gap) / q_result.distance_))
              > angle_tolerance) {
            mimi::utils::PrintDebug(
                "exiting quad loop because angle difference is too big");
            continue;
          }

          // set true to active
          q_result.active_ = true;

          // get traction - t_factor should come out as a negative value
          //
          // technically in literatures, t_factor should be positive
          // and we integrate negative traction.
          // for here, we just keep it as it is, since the results are the same
          //
          // lambda_new = penalty_factor * normal_distance + lambda_old
          const double t_factor =
              nearest_distance_coeff_->coefficient_ * q_normal_gap + q_lagrange;
          // assert(t_factor > 0.0);
          std::cout << "normal is ";
          for (int j{}; j < dim_; ++j) {
            traction_n[j] = t_factor * normal[j];
            std::cout << normal[j] << " ";
          }
          std::cout << std::endl;

          // get sqrt of det of metric tensor
          q_result.query_metric_tensor_weight_ =
              current_quad_derivatives.Weight();

          // get product of all the weights
          // I don't think we need this(* q_target_to_reference_weight)
          const double weight =
              ip.weight * q_result.query_metric_tensor_weight_;

          // set residual
          for (int j{}; j < dim_; ++j) {
            for (int k{}; k < n_dof; ++k) {
              i_residual_view(k, j) += q_shape[k] * traction_n[j] * weight;
            }
          } // res update

        } // quad loop

      } // marked elem loop
    };

    mimi::utils::NThreadExe(assemble_face_residual,
                            marked_boundary_elements_.size(),
                            precomputed_->meshes_.size());
  }

  virtual void AssembleBoundaryGrad(const mfem::Vector& current_x) {
    MIMI_FUNC()
  }

  virtual void AssembleFaceVector(
      const mfem::FiniteElement& element,
      const mfem::FiniteElement&,                   /* unused second element*/
      mfem::FaceElementTransformations& face_trans, /* target to reference */
      const mfem::Vector& current_solution,
      mfem::Vector& residual) {
    MIMI_FUNC()

    const int n_dim = element.GetDim();
    const int n_dof = element.GetDof();

    residual.SetSize(n_dim * n_dof);
    residual = 0.;

    mimi::utils::PrintInfo("ndof",
                           n_dof,
                           "n_dim",
                           n_dim,
                           "not doing anything!");
  }

  virtual void AssembleFaceGrad(
      const mfem::FiniteElement& element,
      const mfem::FiniteElement& /* not used */,
      mfem::FaceElementTransformations& face_trans /* target to reference */,
      const mfem::Vector& input_state /* current solution */,
      mfem::DenseMatrix& jac_res /* stiffness mat */) {

    MIMI_FUNC()

    const int n_dim = element.GetDim();
    const int n_dof = element.GetDof();

    jac_res.SetSize(n_dim * n_dof);
    jac_res = 0.;

    mimi::utils::PrintInfo("ndof",
                           n_dof,
                           "n_dim",
                           n_dim,
                           "not doing anything!");
  }
};

} // namespace mimi::integrators
