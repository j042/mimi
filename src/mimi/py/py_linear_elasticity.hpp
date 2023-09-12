#pragma once

#include <memory>
#include <string>
#include <unordered_map>

/* pybind11 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// splinepy
// #include <splinepy/py/py_spline.hpp>

// mimi
#include "mimi/forms/nonlinear.hpp"
#include "mimi/integrators/linear_elasticity.hpp"
#include "mimi/integrators/penalty_contact.hpp"
#include "mimi/integrators/vector_diffusion.hpp"
#include "mimi/integrators/vector_mass.hpp"
#include "mimi/operators/linear_elasticity.hpp"
#include "mimi/py/py_solid.hpp"
#include "mimi/solvers/newton.hpp"
#include "mimi/utils/lambda_and_mu.hpp"
#include "mimi/utils/precomputed.hpp"

namespace mimi::py {

class PyLinearElasticity : public PySolid {
public:
  double mu_, lambda_, rho_, viscosity_{-1.0};

  using Base_ = PySolid;

  PyLinearElasticity() = default;

  virtual void SetParameters(const double mu,
                             const double lambda,
                             const double rho,
                             const double viscosity = -1.0) {
    mu_ = mu;
    lambda_ = lambda;
    rho_ = rho;
    viscosity_ = viscosity;
  }

  virtual void SetParameters_(const double young,
                              const double poisson,
                              const double rho,
                              const double viscosity = -1.0) {
    auto const& [mu, lambda] = mimi::utils::ToLambdaAndMu(young, poisson);
    SetParameters(mu, lambda, rho, viscosity);
  }

  virtual void Setup(const int nthreads = -1) {
    MIMI_FUNC()

    // selected comments from ex10.
    //  Define the vector finite element spaces representing the mesh
    //  deformation x, the velocity v, and the initial configuration, x_ref.
    //  Since x and v are integrated in time as a system,
    //  we group them together in block vector vx, with offsets given by the
    //  fe_offset array.

    // get fespace - this should be owned by GridFunc
    // and grid func should own the fespace
    // first get nurbs fecollection
    auto* fe_collection = Base_::Mesh()->GetNodes()->OwnFEC();
    if (!fe_collection) {
      mimi::utils::PrintAndThrowError("FE collection does not exist in mesh");
    }

    // create displacement fe space
    auto& disp_fes = Base_::fe_spaces_["displacement"];
    // here, if we pass NURBSext, it will steal the ownership, causing segfault.
    disp_fes.fe_space_ = std::make_unique<mfem::FiniteElementSpace>(
        Base_::Mesh().get(),
        nullptr, // Base_::Mesh()->NURBSext,
        fe_collection,
        MeshDim(),
        // this is how mesh provides its nodes, so solutions should match them
        // else, dofs does not match
        mfem::Ordering::byVDIM);

    // create precomputed data
    auto& bilinear_precomputed = disp_fes.precomputed_["bilinear_forms"];
    bilinear_precomputed = std::make_shared<mimi::utils::PrecomputedData>();
    bilinear_precomputed->Setup(*disp_fes.fe_space_, nthreads);

    // create solution fieds for displacements
    mfem::GridFunction& x = disp_fes.grid_functions_["x"];
    x.SetSpace(disp_fes.fe_space_.get());
    // let's register this ptr to use base class' time stepping
    Base_::x2_ = &x;

    // and velocity
    mfem::GridFunction& v = disp_fes.grid_functions_["x_dot"];
    v.SetSpace(disp_fes.fe_space_.get());
    Base_::x2_dot_ = &v;

    // and reference / initial reference. initialize with fe space then
    // copy from mesh
    mfem::GridFunction& x_ref = disp_fes.grid_functions_["x_ref"];
    x_ref.SetSpace(disp_fes.fe_space_.get());
    x_ref = *Base_::Mesh()->GetNodes();

    mimi::utils::PrintInfo("Setting initial conditions");

    // set initial condition
    x = x_ref;
    v = 0.0;

    /// find bdr dof ids
    Base_::FindBoundaryDofIds();

    // create a linear elasticity operator
    // at first, this is empty
    auto le_oper =
        std::make_unique<mimi::operators::LinearElasticity>(*disp_fes.fe_space_,
                                                            &x_ref);
    mimi::utils::PrintInfo("Created LE oper");

    // now, setup the system.
    // create a dummy
    mfem::SparseMatrix tmp;
    // 1. mass
    // create bilinform
    auto mass = std::make_shared<mfem::BilinearForm>(disp_fes.fe_space_.get());
    le_oper->AddBilinearForm("mass", mass);

    // create density coeff
    auto& rho = Base_::coefficients_["rho"];
    rho = std::make_shared<mfem::ConstantCoefficient>(rho_);

    // create integrator using density
    // this is owned by bilinear form, so do that first before we forget
    auto* mass_integ = new mimi::integrators::VectorMass(rho);
    mass->AddDomainIntegrator(mass_integ);

    // precompute using nthread
    mass_integ->ComputeElementMatrices(*bilinear_precomputed);

    // assemble and remove zero bc entries
    mass->Assemble(1);
    mass->FormSystemMatrix(disp_fes.zero_dofs_, tmp);

    // release some memory
    mass_integ->element_matrices_.reset(); // release saved matrices

    // 2. viscosity
    if (viscosity_ > 0.0) {
      auto visc =
          std::make_shared<mfem::BilinearForm>(disp_fes.fe_space_.get());
      le_oper->AddBilinearForm("viscosity", visc);

      // create viscosity coeff
      auto& visc_coeff = Base_::coefficients_["viscosity"];
      visc_coeff = std::make_shared<mfem::ConstantCoefficient>(viscosity_);

      // create integrator using viscosity
      // and add it to bilinear form
      auto visc_integ = new mimi::integrators::VectorDiffusion(visc_coeff);
      visc->AddDomainIntegrator(visc_integ);

      // nthread assemble
      visc_integ->ComputeElementMatrices(*bilinear_precomputed);

      visc->Assemble(1);
      visc->FormSystemMatrix(disp_fes.zero_dofs_, tmp);

      visc_integ->element_matrices_.reset();
    }

    // 3. Lin elasitiy stiffness
    // start with bilin form
    auto stiffness =
        std::make_shared<mfem::BilinearForm>(disp_fes.fe_space_.get());
    le_oper->AddBilinearForm("stiffness", stiffness);

    // create coeffs
    auto& mu = Base_::coefficients_["mu"];
    mu = std::make_shared<mfem::ConstantCoefficient>(mu_);
    auto& lambda = Base_::coefficients_["lambda"];
    lambda = std::make_shared<mfem::ConstantCoefficient>(lambda_);

    // create integ
    auto stiffness_integ = new mimi::integrators::LinearElasticity(lambda, mu);
    // auto stiffness_integ = new mfem::ElasticityIntegrator(*lambda, *mu);
    stiffness->AddDomainIntegrator(stiffness_integ);

    // nthread assemble
    stiffness_integ->ComputeElementMatrices(*bilinear_precomputed);

    stiffness->Assemble(1);
    stiffness->FormSystemMatrix(disp_fes.zero_dofs_, tmp);

    stiffness_integ->element_matrices_.reset();

    // 4. linear form
    auto rhs = std::make_shared<mfem::LinearForm>(disp_fes.fe_space_.get());
    bool rhs_set{false};
    // assemble body force
    if (const auto& body_force =
            Base_::boundary_conditions_->InitialConfiguration().body_force_;
        body_force.size() != 0) {
      // rhs is set
      rhs_set = true;

      // create coeff
      auto b_force_coeff =
          std::make_shared<mfem::VectorArrayCoefficient>(Base_::MeshDim());
      Base_::vector_coefficients_["body_force"] = b_force_coeff;
      for (auto const& [dim, value] : body_force) {
        b_force_coeff->Set(dim, new mfem::ConstantCoefficient(value));
      }

      // TODO write this integrator
      rhs->AddDomainIntegrator(
          new mfem::VectorDomainLFIntegrator(*b_force_coeff));
    }

    if (const auto& traction =
            Base_::boundary_conditions_->InitialConfiguration().traction_;
        traction.size() != 0) {

      rhs_set = true;

      // create coeff
      auto traction_coeff =
          std::make_shared<mfem::VectorArrayCoefficient>(Base_::MeshDim());
      Base_::vector_coefficients_["traction"] = traction_coeff;
      // we need a temporary container to hold mfem::Vectors per dim
      std::vector<mfem::Vector> traction_per_dim(Base_::MeshDim());
      for (auto& tpd : traction_per_dim) {
        tpd.SetSize(Base_::Mesh()->bdr_attributes.Max());
        tpd = 0.0;
      }

      // apply values at an appropriate location
      for (auto const& [bid, dim_value] : traction) {
        for (auto const& [dim, value] : dim_value) {
          auto& tpd = traction_per_dim[dim];
          tpd(bid) = value;
        }
      }

      // set
      for (int i{}; i < traction_per_dim.size(); ++i) {
        traction_coeff->Set(i,
                            new mfem::PWConstCoefficient(traction_per_dim[i]));
      }

      // add to linear form
      rhs->AddBoundaryIntegrator(
          new mfem::VectorBoundaryLFIntegrator(*traction_coeff));
    }

    if (rhs_set) {
      rhs->Assemble();
      // remove dirichlet nodes
      rhs->SetSubVector(disp_fes.zero_dofs_, 0.0);
      le_oper->AddLinearForm("rhs", rhs);
    }

    // check contact
    if (const auto& contact =
            Base_::boundary_conditions_->CurrentConfiguration().contact_;
        contact.size() != 0) {

      auto contact_precomputed =
          std::make_shared<mimi::utils::PrecomputedData>();
      bilinear_precomputed->PasteCommonTo(contact_precomputed);

      auto nl_form =
          std::make_shared<mimi::forms::Nonlinear>(disp_fes.fe_space_.get());
      le_oper->AddNonlinearForm("contact", nl_form);
      for (const auto& [bid, nd_coeff] : contact) {
        auto contact_integ =
            std::make_shared<mimi::integrators::PenaltyContact>(
                nd_coeff,
                std::to_string(bid),
                contact_precomputed);
        contact_integ->SetBoundaryMarker(&Base_::boundary_markers_[bid]);
        contact_integ->Prepare();

        nl_form->AddBdrFaceIntegrator(contact_integ,
                                      &Base_::boundary_markers_[bid]);
      }
    }

    // setup linear solver
    auto lin_solver = std::make_shared<mfem::UMFPackSolver>();
    Base_::linear_solvers_["linear_elasticity"] = lin_solver;

    // setup a newton solver
    auto newton = std::make_shared<mimi::solvers::LineSearchNewton>();
    // auto newton = std::make_shared<mimi::solvers::Newton>();
    //  auto newton = std::make_shared<mimi::solvers::Newton>();
    Base_::newton_solvers_["linear_elasticity"] = newton;
    le_oper->SetNewtonSolver(newton);

    // basic config. you can change this using ConfigureNewton()
    newton->iterative_mode = false;
    newton->SetOperator(*le_oper);
    newton->SetSolver(*lin_solver);
    newton->SetPrintLevel(mfem::IterativeSolver::PrintLevel()
                              .Warnings()
                              .Errors()
                              .Summary()
                              .FirstAndLast());
    newton->SetRelTol(1e-8);
    newton->SetAbsTol(1e-12);
    newton->SetMaxIter(MeshDim() * 10);

    // ode
    auto gen_alpha =
        std::make_unique<mimi::solvers::GeneralizedAlpha2>(*le_oper);
    gen_alpha->PrintInfo();

    // finally call setup for the operator
    le_oper->Setup();

    // set dynamic system -> transfer ownership
    Base_::SetDynamicSystem2(le_oper.release(), gen_alpha.release());
  }
};

} // namespace mimi::py
