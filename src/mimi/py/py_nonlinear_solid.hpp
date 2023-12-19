#pragma once

#include <memory>

#include "mimi/integrators/materials.hpp"
#include "mimi/integrators/nonlinear_solid.hpp"
#include "mimi/integrators/penalty_contact.hpp"
#include "mimi/integrators/vector_diffusion.hpp"
#include "mimi/integrators/vector_mass.hpp"
#include "mimi/operators/nonlinear_solid.hpp"
#include "mimi/py/py_solid.hpp"

namespace mimi::py {

class PyNonlinearSolid : public PySolid {
public:
  using Base_ = PySolid;
  using MaterialPointer_ = std::shared_ptr<mimi::integrators::MaterialBase>;

  MaterialPointer_ material_;

  PyNonlinearSolid() = default;

  virtual void SetMaterial(const MaterialPointer_& material) {
    MIMI_FUNC()
    material_ = material;
  }

  virtual void Setup(const int nthreads = -1) {
    MIMI_FUNC()

    // quick 0 and -1 filtering
    const int n_threads = (nthreads < 1) ? 1 : nthreads;

    // get fespace - this should be owned by GridFunc
    // and grid func should own the fespace
    // first get nurbs fecollection
    auto* fe_collection = Base_::Mesh()->GetNodes()->OwnFEC();
    if (!fe_collection) {
      mimi::utils::PrintAndThrowError("FE collection does not exist in mesh");
    }

    // create displacement fe space
    // I realize, this is misleading: we work with x, not u
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

    // we need to create precomputed data one for:
    // - bilinear forms
    // - nonlinear solid
    // - (optional) conatct
    // here, we only create one for bilinear. others are done below
    // note, for bilinear, nothing is really saved. just used for thread safety
    auto& bilinear_precomputed = disp_fes.precomputed_["bilinear_forms"];
    bilinear_precomputed = std::make_shared<mimi::utils::PrecomputedData>();
    bilinear_precomputed->Setup(*disp_fes.fe_space_, n_threads);

    // create solution fields for x and set local reference
    mfem::GridFunction& x = disp_fes.grid_functions_["x"];
    x.SetSpace(disp_fes.fe_space_.get());
    Base_::x2_ = &x; // "2" means x for ode second order

    // and x_dot (= v)
    mfem::GridFunction& x_dot = disp_fes.grid_functions_["x_dot"];
    x_dot.SetSpace(disp_fes.fe_space_.get());
    Base_::x2_dot_ = &x_dot;

    // and reference / initial reference. initialize with fe space then
    // copy from mesh
    mfem::GridFunction& x_ref = disp_fes.grid_functions_["x_ref"];
    x_ref.SetSpace(disp_fes.fe_space_.get());
    x_ref = *Base_::Mesh()->GetNodes();

    // set initial condition
    // you can change this in python using SolutionView()
    x = x_ref;
    x_dot = 0.0;

    // let's process boundaries
    Base_::FindBoundaryDofIds();

    // creating operators
    auto nl_oper =
        std::make_unique<mimi::operators::NonlinearSolid>(*disp_fes.fe_space_,
                                                          &x_ref);

    // let's setup the system
    // create dummy sparse matrix - mfem just sets referencee, so no copies are
    // made
    mfem::SparseMatrix tmp;

    // 1. mass
    auto mass = std::make_shared<mfem::BilinearForm>(disp_fes.fe_space_.get());
    nl_oper->AddBilinearForm("mass", mass);

    // create density coeff for mass integrator
    assert(material_->density_ > 0.0);
    auto& density = Base_::coefficients_["density"];
    density = std::make_shared<mfem::ConstantCoefficient>(material_->density_);

    // create integrator using density
    // this is owned by bilinear form, so do that first before we forget
    auto* mass_integ = new mimi::integrators::VectorMass(density);
    mass->AddDomainIntegrator(mass_integ);

    // precompute using nthread
    mass_integ->ComputeElementMatrices(*bilinear_precomputed);

    // assemble and form matrix with bc in mind
    mass->Assemble(0);
    mass->FormSystemMatrix(disp_fes.zero_dofs_, tmp);
    mass_integ->element_matrices_.reset(); // release saved matrices

    // 2. damping / viscosity, as mfem calls it
    if (material_->viscosity_ > 0.0) {
      auto visc =
          std::make_shared<mfem::BilinearForm>(disp_fes.fe_space_.get());
      nl_oper->AddBilinearForm("viscosity", visc);

      // create viscosity coeff
      auto& visc_coeff = Base_::coefficients_["viscosity"];
      visc_coeff =
          std::make_shared<mfem::ConstantCoefficient>(material_->viscosity_);

      // create integrator using viscosity
      // and add it to bilinear form
      auto visc_integ = new mimi::integrators::VectorDiffusion(visc_coeff);
      visc->AddDomainIntegrator(visc_integ);

      // nthread assemble
      visc_integ->ComputeElementMatrices(*bilinear_precomputed);
      visc->Assemble(0);
      visc->FormSystemMatrix(disp_fes.zero_dofs_, tmp);
      visc_integ->element_matrices_.reset();
    }

    // 3. nonlinear stiffness
    // first pre-computed
    auto& nonlinear_stiffness_precomputed =
        disp_fes.precomputed_["nonlinear_stiffness"];
    nonlinear_stiffness_precomputed =
        std::make_shared<mimi::utils::PrecomputedData>();
    bilinear_precomputed->PasteCommonTo(nonlinear_stiffness_precomputed);

    // nlform
    auto nonlinear_stiffness =
        std::make_shared<mimi::forms::Nonlinear>(disp_fes.fe_space_.get());
    // add it to operator
    nl_oper->AddNonlinearForm("nonlinear_stiffness", nonlinear_stiffness);
    // create integrator
    auto nonlinear_solid_integ =
        std::make_shared<mimi::integrators::NonlinearSolid>(
            material_->Name() + "-NonlinearSolid",
            material_,
            nonlinear_stiffness_precomputed);
    nonlinear_solid_integ->Prepare();
    // add integrator to nl form
    nonlinear_stiffness->AddDomainIntegrator(nonlinear_solid_integ);
    nonlinear_stiffness->SetEssentialTrueDofs(disp_fes.zero_dofs_);

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
      nl_oper->AddLinearForm("rhs", rhs);
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
      nl_oper->AddNonlinearForm("contact", nl_form);
      for (const auto& [bid, nd_coeff] : contact) {
        // initialzie integrator with nearest distance coeff (splinepy splines)
        auto contact_integ =
            std::make_shared<mimi::integrators::PenaltyContact>(
                nd_coeff,
                std::to_string(bid),
                contact_precomputed);
        // set the same boundary marker. It will be internally used for nthread
        // assemble of marked boundaries
        contact_integ->SetBoundaryMarker(&Base_::boundary_markers_[bid]);
        // precompute basis and recurring options.
        contact_integ->Prepare();

        nl_form->AddBdrFaceIntegrator(contact_integ,
                                      &Base_::boundary_markers_[bid]);
      }
    }

    // setup solvers
    // setup linear solver - should try SUNDIALS::SuperLU_mt
    auto lin_solver = std::make_shared<mfem::UMFPackSolver>();
    Base_::linear_solvers_["nonlinear_solid"] = lin_solver;

    // setup a newton solver
    auto newton = std::make_shared<mimi::solvers::LineSearchNewton>();
    // give pointer of nl oper to control line search assembly
    newton->nl_oper_ = nl_oper.get();
    Base_::newton_solvers_["nonlinear_solid"] = newton;
    // basic config. you can change this using ConfigureNewton()
    newton->iterative_mode = false;
    newton->SetOperator(*nl_oper);
    newton->SetSolver(*lin_solver);
    newton->SetPrintLevel(mfem::IterativeSolver::PrintLevel()
                              .Warnings()
                              .Errors()
                              .Summary()
                              .FirstAndLast());
    newton->SetRelTol(1e-8);
    newton->SetAbsTol(1e-12);
    newton->SetMaxIter(MeshDim() * 10);

    // finally, register newton solver to the opeartor
    nl_oper->SetNewtonSolver(newton);

    // ode
    auto gen_alpha =
        std::make_unique<mimi::solvers::GeneralizedAlpha2>(*nl_oper);
    gen_alpha->PrintInfo();

    // finalize operator
    nl_oper->Setup();

    // set dynamic system -> transfer ownership of those to base
    Base_::SetDynamicSystem2(nl_oper.release(), gen_alpha.release());
  }
};

} // namespace mimi::py
