#pragma once

#include <cmath>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/coefficients/nearest_distance.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

class PenaltyContact : public NonlinearBase {
protected:
  /// scene
  std::shared_ptr<mimi::coefficients::NearestDistanceBase>
      nearest_distance_coeff_ = nullptr;

  /// marker for this bdr face integ
  const mfem::Array<int>* boundary_marker_ = nullptr;

  /// ids of contributing boundary elements, extracted based on boundary_marker_
  mimi::utils::Vector<int> marked_boundary_elements_;

  /// @brief quadrature order per elements - alternatively, could be per patch
  /// thing
  mimi::utils::Vector<int> quadrature_orders_;

  /// results from proximity queries, to be reused for grad - latest relevant
  /// residual and grad are calls right after one another from newton solver
  mimi::utils::Vector<
      mimi::utils::Vector<mimi::coefficients::NearestDistanceBase::Results>>
      nearest_distance_results_;

  /// decided to ask for intrules all the time. but since we don't wanna call
  /// the geometry type all the time, we save this just once.
  mfem::Geometry::Type boundary_geometry_type_;

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
    const int n_boundary_patches = precomputed_->meshes_[0]->NURBSext->GetNBP();
    const int n_boundary_elements =
        precomputed_->meshes_[0]->NURBSext->GetNBE();
    const int n_threads =
        precomputed_->meshes_.size(); // size of mesh is nthread from Setup()

    // get marked boundary elements: this reduces size of the loop
    if (!boundary_marker_) {
      mimi::utils::PrintAndThrowError(Name(),
                                      "does not have a boundary marker.");
    }
    // prepare loop - deref and get a mesh to ask
    auto& mesh = *precomputed_->meshes_[0];
    auto& b_marker = *boundary_marker_;
    marked_boundary_elements_.reserve(n_boundary_elements);
    // loop boundary elements
    for (int i{}; i < n_boundary_elements; ++i) {
      auto& b_el = precomputed_->boundary_elements_[i];
      const int bdr_attr = mesh.GetBdrAttribute(i);
      if (b_marker[bdr_attr - 1] == 0) {
        continue;
      }
      marked_boundary_elements_.push_back(i);
    }
    marked_boundary_elements_.shrink_to_fit();

    // extract boundary geometry type
    boundary_geometry_type_ =
        precomputed_->boundary_elements_[0]->GetGeomType();

    // allocate result holder
    nearest_distance_results_.resize(n_boundary_elements);

    // allocate element vectors and matrices
    Base_::element_matrices_->SetSize(n_boundary_elements);
    Base_::element_vectors_->SetSize(n_boundary_elements);

    // shapes
    auto& boundary_shapes = precomputed_->vectors_["boundary_shapes"];
    boundary_shapes.resize(n_boundary_patches);

    // jacobian weights
    auto& boundary_target_to_reference_weights =
        precomputed_->scalars_["boundary_target_to_reference_weights"];
    boundary_target_to_reference_weights.resize(n_boundary_elements);

    // quad order
    quadratue_orders_.resize(n_boundary_elements);

    // shapes is once per boundary patch. yes, this will have many recurring
    // shapes, which makes it easy to lookup
    auto precompute_shapes = [&](const int b_patch_begin,
                                 const int b_patch_end,
                                 const int i_thread) {
      // thread's obj
      auto& int_rules = precomputed_->int_rules_[i_thread];

      for (int i{b_patch_begin}; i < b_patch_end; ++i) {
        // get corresponding vector of shapes
        auto& shapes = boundary_shapes[i];

        // get id of a patch
        const int b_el_id =
            precomputed_->meshes_[0]->NURBSext->GetPatchBdrElements(i)[0];

        // get elem
        auto& b_el = *precomputed_->boundary_elements_[b_el_id];
        const int n_dof = b_el.GetDof();

        // get quad order
        const int q_order =
            (quadrature_order < 0) ? b_el.GetOrder() * 2 + 3 : quadrature_order;

        // get int rule
        const mfem::IntegrationRule& ir =
            int_rules.Get(boundary_geometry_type_, q_order);
        // prepare quad loop
        const int n_quad = ir.GetNPoints();
        shapes.resize(n_quad);

        // quad loop - calc shapes
        for (int j{}; j < n_quad; ++j) {
          mfem::Vector& shape = shapes[j];
          shape.SetSize(n_dof);

          const mfem::IntegrationPoint& ip = ir.IntPoint(j);
          b_el.CalcShape(ip, shape);
        }
      }
    };

    // now, weight of jacobian.
    auto precompute_trans_weights = [&](const int marked_b_el_begin,
                                        const int marked_b_el_end,
                                        const int i_thread) {
      // thread's obj
      auto& int_rules = precomputed_->int_rules_[i_thread];

      for (int i{marked_b_el_begin}; i < marked_b_el_end; ++i) {
        const int& i_mbe = marked_boundary_elements_[i];
        // get boundary transformations
        auto& b_trans =
            *precomputed_->target_to_reference_boundary_trans_[i_mbe];
        // get corresponding vector of weights
        auto& weights = boundary_target_to_reference_weights[i_mbe];
        // get results holder to allocate
        auto& results = nearest_distance_results_[i_mbe];
        // get quad order
        const int q_order = (quadrature_order < 0)
                                ? b_trans.GetFE()->GetOrder() * 2 + 3
                                : quadrature_order;

        // save this quad_order for the element
        quadrature_orders_[i_mbe] = q_order;

        // get int rule
        const mfem::IntegrationRule& ir =
            int_rules.Get(boundary_geometry_type_, q_order);
        // prepare quad loop
        const int n_quad = ir.GetNPoints();
        // use n_quad to allocate
        weights.resize(n_quad);
        results.resize(n_quad);

        for (int j{}; j < n_quad; ++j) {
          const mfem::IntegrationPoint& ip = ir.IntPoint(j);
          b_trans.SetIntPoint(&ip);
          weights[j] = b_trans.Weight();

          // alloate each members of results
          results[j].SetSize(b_trans.GetSpaceDim() - 1, b_trans.GetSpaceDim());
        }
      }
    };

    mimi::utils::NThreadExe(precompute_shapes, n_boundary_patches, n_threads);
    mimi::utils::NThreadExe(precompute_trans_weights,
                            static_cast<int>(marked_boundary_elements_.size()),
                            n_threads);
  }

  virtual void AssembleFaceResidual(const mfem::Vector& current_x) {
    MIMI_FUNC()

    // get related precomputed values
    auto& boundary_shapes =
        precomputed_->vectors_["boundary_shapes"]; // n_patches * n_quad * n_dof
                                                   // - each can vary in size

    // jacobian weights
    auto& boundary_target_to_reference_weights =
        precomputed_
            ->scalars_["boundary_target_to_reference_weights"]; // n_b_elem *
                                                                // n_quad

    auto assemble_face_residual = [&](const int begin,
                                      const int end,
                                      const int i_thread) {
      // prepare per thread data - query, int_rule
      mimi::coefficients::NearestDistanceBase::Query query{};
      query.max_iterations_ = 20;
      query.query_.SetSize(precomputed_->meshes_[0]->Dimension()); // allocate

      auto& int_rules = precomputed_->int_rules_[i_thread];

      // this loops marked boundary elements
      for (int i{begin}; i < end; ++i) {
        // get this loop's objects
        const int i_mbe = marked_boundary_elements_[i];
        const auto& i_b_vdof = precomputed_->boundary_v_dofs_[i_mbe];
        auto& i_b_el = precomputed_->boundary_elements_[i_mbe];
        const auto& i_shapes = boundary_shapes[i_b_el->GetPatch()];
        auto& i_results = nearest_distance_results_[i_mbe];

        // sizes
        const int n_dim = i_b_el->GetDim();
        const int n_dof = i_b_el->GetDof();

        // copy current solution
        mfem::Vector i_current_solution_vec;
        current_x.GetSubVector(i_b_vdof, i_current_solution_vec);
        mfem::DenseMatrix i_current_solution(i_current_solution_vec.GetData(),
                                             n_dof,
                                             n_dim);

        // prepare quad loop
        const auto& int_rule =
            int_rules.Get(boundary_geometry_type_, quadrature_orders[i_mbe]);

        // quad loop
        for (int q{}; q < int_rule.GetNPoints(); ++q) {
          auto& q_shape = i_shapes[q];
          auto& q_result = i_results[q];
          // formulate query
          query.query_.fill(0.0);
          for (int j{}; j < n_dof; ++j) {
            for (int k{}; k < n_dim; ++k) {
              query.query_[k] += q_shape(j) * i_current_solution(j, k);
            }
          }

          // query nearest distance
          nearest_distance_coeff_->NearestDistance(query, q_result);
        }
      }
    }
  }

  virtual void AssembleFaceResidualGrad(const mfem::Vector& current_x) {
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

    mfem::Vector el_shape(n_dof);

    mfem::IntegrationRules& int_rules = precomputed_->int_rules_[0]; // i_thread

    mimi::utils::PrintInfo("GeomTrype",
                           face_trans.GetGeometryType(),
                           face_trans.GetFE()->GetGeomType());

    const mfem::IntegrationRule& ir =
        /// int_rules.Get(face_trans.GetGeometryType(), 3);
        int_rules.Get(face_trans.GetFE()->GetGeomType(), 3);
    const mfem::IntegrationPoint& ip = ir.IntPoint(0);
    face_trans.SetIntPoint(&ip);
    const mfem::IntegrationPoint& eip = face_trans.GetElement1IntPoint();

    element.CalcShape(eip, el_shape);

    auto& bel = *face_trans.GetFE();
    mfem::Vector bel_shape(bel.GetDof());

    bel.CalcShape(ip, bel_shape);

    mimi::utils::PrintInfo("element ndof", n_dof);
    for (int i{}; i < n_dof; ++i) {
      std::cout << el_shape[i] << " ";
    }
    auto& jac = face_trans.Elem1->Jacobian();
    // std::cout << "\n jacobian det " << face_trans.Face->Weight() << "\n";
    std::cout << "\n jacobian det " << jac.Weight() << "\n";
    for (int i{}; i < jac.Height(); ++i) {
      for (int j{}; j < jac.Width(); ++j) {
        std::cout << jac(i, j) << " ";
      }
      std::cout << "\n";
    }
    std::cout << "element_no: " << face_trans.Elem1No << "patch_no: "
              << precomputed_->elements_[face_trans.Elem1No]->GetPatch()
              << "\n";
    std::cout << "element_vdof\n";
    for (int i{}; i < precomputed_->v_dofs_[face_trans.Elem1No]->Size(); ++i) {
      std::cout << (*precomputed_->v_dofs_[face_trans.Elem1No])[i] << " ";
    }
    std::cout << "\n";

    mimi::utils::PrintInfo("b element ndof", bel.GetDof());
    for (int i{}; i < bel.GetDof(); ++i) {
      std::cout << bel_shape[i] << " ";
    }
    std::cout << "\n jacobian det " << face_trans.Weight() << "\n";
    auto& jac2 = face_trans.Face->Jacobian();
    for (int i{}; i < jac2.Height(); ++i) {
      for (int j{}; j < jac2.Width(); ++j) {
        std::cout << jac2(i, j) << " ";
      }
      std::cout << "\n";
    }
    std::cout
        << "bdr element_no: " << face_trans.ElementNo << "patch_no: "
        << precomputed_->boundary_elements_[face_trans.ElementNo]->GetPatch()
        << "\n";
    std::cout << "bdr element_vdof\n";
    for (int i{};
         i < precomputed_->boundary_v_dofs_[face_trans.ElementNo]->Size();
         ++i) {
      std::cout << (*precomputed_->boundary_v_dofs_[face_trans.ElementNo])[i]
                << " ";
    }
    std::cout << "\n";

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
