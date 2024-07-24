// mimi
#include "mimi/py/py_nonlinear_solid.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_nonlinear_solid(py::module_& m) {
  py::class_<PyNonlinearSolid, std::shared_ptr<PyNonlinearSolid>, PySolid>
      klasse(m, "NonlinearSolid");
  klasse.def(py::init<>())
      .def("set_material", &PyNonlinearSolid::SetMaterial, py::arg("material"));
}

void PyNonlinearSolid::Setup(const int nthreads) {
  MIMI_FUNC()

  // quick 0 and -1 filtering
  const int n_threads = (nthreads < 1) ? 1 : nthreads;

  SetupNTheads(n_threads);

  // get fespace - this should be owned by GridFunc
  // and grid func should own the fespace
  // first get nurbs fecollection
  auto* fe_collection = Base_::Mesh()->GetNodes()->OwnFEC();
  if (!fe_collection) {
    mimi::utils::PrintAndThrowError("FE collection does not exist in mesh");
  }

  // create displacement fe space
  auto& disp_fes = Base_::fe_spaces_["displacement"];

  // check if we have periodic bc. if so, we need to create everything again
  if (auto& periodic_map = Base_::boundary_conditions_->InitialConfiguration()
                               .periodic_boundaries_;
      !periodic_map.empty()) {
    mimi::utils::PrintInfo("Periodic boundaries requested.");

    const int n_periodic = periodic_map.size();
    mfem::Array<int> b0s(n_periodic), b1s(n_periodic);
    b0s = -1;
    b1s = -1;
    int i_p{};
    for (const auto& [bid0, bid1] : periodic_map) {
      mimi::utils::PrintInfo("Requested to connect", bid0, bid1);
      b0s[i_p] = bid0;
      b1s[i_p] = bid1;
      ++i_p;
    }
    // following nurbs_ex1, create a separate ext
    auto nurbsext =
        std::make_unique<mfem::NURBSExtension>(*Base_::Mesh()->NURBSext);
    nurbsext->ConnectBoundaries(b0s, b1s);

    // create new fe space with this nurbsext
    disp_fes.fe_space =
        std::make_unique<mfem::FiniteElementSpace>(Base_::Mesh().get(),
                                                   // takes ownership
                                                   nurbsext.release(),
                                                   fe_collection,
                                                   MeshDim(),
                                                   mfem::Ordering::byVDIM);
  } else {
    // here, if we pass NURBSext, it will steal the ownership, causing
    // segfault.
    disp_fes.fe_space = std::make_unique<mfem::FiniteElementSpace>(
        Base_::Mesh().get(),
        nullptr, // Base_::Mesh()->NURBSext,
        fe_collection,
        MeshDim(),
        // this is how mesh provides its nodes, so solutions should match them
        // else, dofs does not match
        mfem::Ordering::byVDIM);
  }

  mimi::utils::PrintInfo("Creating grid functions u");
  // create solution fields for x and set local reference
  mfem::GridFunction& x = disp_fes.grid_functions["x"];
  x.SetSpace(disp_fes.fe_space.get());
  Base_::x2_ = &x; // "2" means x for ode second order

  // and x_dot (= v)
  mimi::utils::PrintInfo("Creating grid functions v");
  mfem::GridFunction& x_dot = disp_fes.grid_functions["x_dot"];
  x_dot.SetSpace(disp_fes.fe_space.get());
  Base_::x2_dot_ = &x_dot;

  // and reference / initial reference. initialize with fe space then
  // copy from mesh
  mimi::utils::PrintInfo("Creating grid functions x_ref");
  mfem::GridFunction& x_ref = disp_fes.grid_functions["x_ref"];
  x_ref.SetSpace(disp_fes.fe_space.get());

  // carefully create x_ref, in case we have periodic mesh
  mfem::GridFunction& nodes = *Base_::Mesh()->GetNodes();
  mimi::utils::PrintInfo("Number of geometric coordinates x dim:",
                         nodes.Size());
  mimi::utils::PrintInfo("Size of grid function:", x_ref.Size());

  // if there's more efficient way, do so.
  if (nodes.Size() != x_ref.Size()) {
    mfem::NURBSExtension& ext = *disp_fes.fe_space->GetNURBSext();
    const int dim = Base_::MeshDim();
    const double* node_d = nodes.GetData();
    for (int i{}; i < nodes.Size() / dim; ++i) {
      const int k = ext.DofMap(i);
      for (int j{}; j < dim; ++j) {
        x_ref[k * dim + j] = *node_d++;
      }
    }
  } else {
    // they should have same fespace
    x_ref = nodes;
  }

  // set initial condition
  // you can change this in python using SolutionView()
  x = 0.0; // this is u
  x_dot = 0.0;

  // we need to create precomputed data one for:
  // - bilinear forms
  // - nonlinear solid
  // - (optional) conatct
  // here, we only create one for bilinear. others are done below
  // note, for bilinear, nothing is really saved. just used for thread safety
  mimi::utils::PrintDebug("Creating PrecomputedData");
  // create precomputed for this FESpace
  disp_fes.precomputed = std::make_shared<mimi::utils::PrecomputedData>();

  mimi::utils::PrintDebug("Setup PrecomputedData");
  disp_fes.precomputed->PrepareThreadSafety(*disp_fes.fe_space, n_threads);
  disp_fes.precomputed->PrepareElementData();
  disp_fes.precomputed->PrepareSparsity();
  // let's process boundaries
  mimi::utils::PrintDebug("Find boundary dof ids");
  Base_::FindBoundaryDofIds();

  // create element data - simple domain precompute
  disp_fes.precomputed->CreateElementQuadData("domain", nullptr);
  // once we have parallel bilinear form assembly we can re directly cal;
  // precompute

  // creating operators
  auto nl_oper =
      std::make_unique<mimi::operators::NonlinearSolid>(*disp_fes.fe_space);

  // let's setup the system
  // create dummy sparse matrix - mfem just sets referencee, so no copies are
  // made
  mfem::SparseMatrix tmp;

  // 1. mass
  auto mass = std::make_shared<mfem::BilinearForm>(disp_fes.fe_space.get());
  nl_oper->AddBilinearForm("mass", mass);

  // create density coeff for mass integrator
  assert(material_->density_ > 0.0);

  // temporarily disable custom parallel integrators
  // TODO: task force!
  // auto& density = Base_::coefficients_["density"];
  // density =
  // std::make_shared<mfem::ConstantCoefficient>(material_->density_); auto*
  // mass_integ = new mimi::integrators::VectorMass(density);

  // create integrator using density
  // this is owned by bilinear form, so do that first before we forget
  mfem::ConstantCoefficient rho(material_->density_);
  mass->AddDomainIntegrator(new mfem::VectorMassIntegrator(rho));
  mass->Assemble(0);
  mass->FormSystemMatrix(disp_fes.zero_dofs, tmp);

  // 2. damping / viscosity, as mfem calls it
  if (material_->viscosity_ > 0.0) {
    auto visc = std::make_shared<mfem::BilinearForm>(disp_fes.fe_space.get());
    nl_oper->AddBilinearForm("viscosity", visc);

    // create viscosity coeff
    // auto& visc_coeff = Base_::coefficients_["viscosity"];
    // visc_coeff =
    // std::make_shared<mfem::ConstantCoefficient>(material_->viscosity_);
    // auto visc_integ = new mimi::integrators::VectorDiffusion(visc_coeff);

    // create integrator using viscosity
    // and add it to bilinear form
    mfem::ConstantCoefficient v_coef(material_->viscosity_);
    visc->AddDomainIntegrator(new mfem::VectorDiffusionIntegrator(v_coef));
    visc->Assemble(0);
    visc->FormSystemMatrix(disp_fes.zero_dofs, tmp);
  }

  // 3. nonlinear stiffness
  // first pre-computed
  // nlform
  auto nonlinear_stiffness =
      std::make_shared<mimi::forms::Nonlinear>(disp_fes.fe_space.get());
  // add it to operator
  nl_oper->AddNonlinearForm("nonlinear_stiffness", nonlinear_stiffness);
  // create integrator
  auto nonlinear_solid_integ =
      std::make_shared<mimi::integrators::NonlinearSolid>(
          material_->Name() + "-NonlinearSolid",
          material_,
          disp_fes.precomputed);
  nonlinear_solid_integ->runtime_communication_ = RuntimeCommunication();
  // we precompute for nonlinear solid
  const int default_solid_q =
      RuntimeCommunication()->GetInteger("nonlinear_solid_quadrature_order",
                                         -1);
  disp_fes.precomputed->PrecomputeElementQuadData("domain",
                                                  default_solid_q,
                                                  true);
  nonlinear_solid_integ->Prepare();
  // add integrator to nl form
  nonlinear_stiffness->AddDomainIntegrator(nonlinear_solid_integ);
  nonlinear_stiffness->SetEssentialTrueDofs(disp_fes.zero_dofs);

  // 4. linear form
  auto rhs = std::make_shared<mfem::LinearForm>(disp_fes.fe_space.get());
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
    for (int i{}; i < static_cast<int>(traction_per_dim.size()); ++i) {
      traction_coeff->Set(i, new mfem::PWConstCoefficient(traction_per_dim[i]));
    }

    // add to linear form
    rhs->AddBoundaryIntegrator(
        new mfem::VectorBoundaryLFIntegrator(*traction_coeff));
  }

  if (rhs_set) {
    rhs->Assemble();
    // remove dirichlet nodes
    rhs->SetSubVector(disp_fes.zero_dofs, 0.0);
    nl_oper->AddLinearForm("rhs", rhs);
  }

  // check contact
  if (const auto& contact =
          Base_::boundary_conditions_->CurrentConfiguration().contact_;
      contact.size() != 0) {

    auto nl_form =
        std::make_shared<mimi::forms::Nonlinear>(disp_fes.fe_space.get());
    nl_oper->AddNonlinearForm("contact", nl_form);
    for (const auto& [bid, nd_coeff] : contact) {
      // initialzie integrator with nearest distance coeff (splinepy splines)
      auto contact_integ = std::make_shared<mimi::integrators::MortarContact>(
          nd_coeff,
          "contact" + std::to_string(bid),
          disp_fes.precomputed);

      contact_integ->runtime_communication_ = RuntimeCommunication();

      // let's get marked boundaries to create EQData
      contact_integ->SetBoundaryMarker(&Base_::boundary_markers_[bid]);
      mimi::utils::Vector<int>& marked_bdr =
          contact_integ->MarkedBoundaryElements();

      // precompute
      disp_fes.precomputed->CreateBoundaryElementQuadData(
          contact_integ->Name(),
          marked_bdr.data(),
          static_cast<int>(marked_bdr.size()));
      const int default_contact_q =
          RuntimeCommunication()->GetInteger("contact_quadrature_order", -1);
      disp_fes.precomputed->PrecomputeElementQuadData(contact_integ->Name(),
                                                      default_contact_q,
                                                      false /* dNdX*/);

      // set the same boundary marker. It will be internally used for nthread
      // assemble of marked boundaries
      // precompute basis and recurring options.
      contact_integ->Prepare();

      nl_form->AddBdrFaceIntegrator(contact_integ,
                                    &Base_::boundary_markers_[bid]);
    }
  }

  // setup solvers
  // setup linear solver - should try SUNDIALS::SuperLU_mt
  if (RuntimeCommunication()->GetInteger("use_iterative_solver", 0)) {
    auto lin_solver = std::make_shared<mfem::GMRESSolver>();
    auto prec = std::make_shared<mfem::DSmoother>();
    lin_solver->SetRelTol(1e-8);
    lin_solver->SetAbsTol(1e-12);
    lin_solver->SetMaxIter(300);
    lin_solver->SetPrintLevel(-1);
    lin_solver->SetPreconditioner(*prec);
    Base_::linear_solvers_["nonlinear_solid"] = lin_solver;
    Base_::linear_solvers_["nonlinear_solid_preconditioner"] = prec;
  } else {
    auto lin_solver = std::make_shared<mfem::UMFPackSolver>();
    Base_::linear_solvers_["nonlinear_solid"] = lin_solver;
  }

  // setup a newton solver
  auto newton = std::make_shared<mimi::solvers::LineSearchNewton>();
  // give pointer of nl oper to control line search assembly
  newton->mimi_oper_ = nl_oper.get();
  Base_::newton_solvers_["nonlinear_solid"] = newton;
  // basic config. you can change this using ConfigureNewton()
  newton->iterative_mode = false;
  newton->SetOperator(*nl_oper);
  newton->SetSolver(*Base_::linear_solvers_["nonlinear_solid"]);
  newton->SetPrintLevel(mfem::IterativeSolver::PrintLevel()
                            .Warnings()
                            .Errors()
                            .Summary()
                            .FirstAndLast());
  newton->SetRelTol(1e-8);
  newton->SetAbsTol(1e-12);
  newton->SetMaxIter(MeshDim() * 10);

  // ode
  const double rho_inf =
      RuntimeCommunication()->GetReal("ode_coefficient", .25);
  auto odesolver =
      std::make_unique<mimi::solvers::GeneralizedAlpha2>(*nl_oper, rho_inf);
  odesolver->PrintInfo();
  if (!Base_::boundary_conditions_->InitialConfiguration()
           .constant_velocity_.empty()) {
    auto dynamic_dirichlet = std::make_shared<
        mimi::utils::TimeDependentDirichletBoundaryCondition>();
    dynamic_dirichlet->boundary_dof_ids_ = &disp_fes.boundary_dof_ids;
    dynamic_dirichlet->dynamic_bc_ = Base_::boundary_conditions_.get();
    odesolver->dynamic_dirichlet_ = dynamic_dirichlet;
  }
  Base_::boundary_conditions_->Print();

  // finalize operator
  nl_oper->Setup();

  // set dynamic system -> transfer ownership of those to base
  Base_::SetDynamicSystem2(nl_oper.release(), odesolver.release());
}

} // namespace mimi::py
