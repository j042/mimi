// mimi
#include "mimi/py/py_stokes.hpp"

namespace mimi::py {

namespace py = pybind11;

void init_py_stokes(py::module_& m) {
  py::class_<PyStokes,
             std::shared_ptr<PyStokes>,
             PySolid>
      klasse(m, "Stokes");
  klasse.def(py::init<>())
      .def("set_material",
           &PyStokes::SetMaterial,
           py::arg("material"));
}

void PyStokes::Setup(const int nthreads {
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
    auto& velocity_fes = Base_::fe_spaces_["velocity"];
    auto& pressure_fes = Base_::fe_spaces_["pressure"];

    // copy mesh for velocity
    vel_mesh_ = std::make_unique<mfem::Mesh>(*Base_::mesh_, true);
    vel_mesh_->DegreeElevate(1, 50);

    // check if we have periodic bc. if so, we need to create everything again
    if (auto& periodic_map = Base_::boundary_conditions_->InitialConfiguration()
                                 .periodic_boundaries_;
        !periodic_map.empty()) {
      mimi::utils::PrintAndThrowError(
          "Periodic boundaries requested, but not implemented");
    } else {
      // here, if we pass NURBSext, it will steal the ownership, causing
      // segfault.
      velocity_fes.fe_space_ = std::make_unique<mfem::FiniteElementSpace>(
          vel_mesh_.get(),
          nullptr,
          fe_collection,
          MeshDim(),
          // this is how mesh provides its nodes, so solutions should match them
          // else, dofs does not match
          mfem::Ordering::byVDIM);
      pressure_fes.fe_space_ =
          std::make_unique<mfem::FiniteElementSpace>(Base_::Mesh().get(),
                                                     nullptr,
                                                     fe_collection,
                                                     MeshDim(),
                                                     mfem::Ordering::byVDIM);
    }

    // we need to create precomputed data one for:
    // - bilinear forms
    // - nonlinear solid
    // here, we only create one for bilinear. others are done below
    // note, for bilinear, nothing is really saved. just used for thread safety
    mimi::utils::PrintDebug("Creating PrecomputedData");
    velocity_fes.precomputed_ =
        std::make_shared<mimi::utils::PrecomputedData>();
    pressure_fes.precomputed_ =
        std::make_shared<mimi::utils::PrecomputedData>();

    mimi::utils::PrintDebug("Setup PrecomputedData");
    // first compute sparcity of each block
    velocity_fes.precomputed_->PrepareThreadSafety(*velocity_fes.fe_space_,
                                                   n_threads);
    velocity_fes.precomputed_->PrepareElementData();
    pressure_fes.precomputed_->PrepareThreadSafety(*pressure_fes.fe_space_,
                                                   n_threads);
    pressure_fes.precomputed_->PrepareElementData();
    // let's process boundaries
    mimi::utils::PrintDebug("Find boundary dof ids");
    Base_::FindBoundaryDofIds();

    // create element data - simple domain precompute
    const int vel_order = velocity_fes.fe_space_->GetMaxElementOrder();
    const int default_stokes_q =
        RuntimeCommunication()->GetInteger("stokes_quadrature_order",
                                           2 * vel_order + 3);
    velocity_fes.precomputed_->CreateElementQuadData("domain", nullptr);
    // get N, dN_dxi, dN_dX
    velocity_fes.precomputed_->PrecomputeElementQuadData("domain",
                                                         default_stokes_q,
                                                         true);
    pressure_fes.precomputed_->CreateElementQuadData("domain", nullptr);
    // get N, dN_dxi - we don't need dN_dxi, but for shape optimization loop, we
    // can use that to compute jacobian of the changed shape
    pressure_fes.precomputed_->PrecomputeElementQuadData("domain",
                                                         default_stokes_q,
                                                         false);

    // once we have parallel bilinear form assembly we can re directly cal;
    // precompute

    mimi::utils::PrintInfo("Creating grid functions v");
    // create solution fields for x and set local reference
    mfem::GridFunction& v = velocity_fes.grid_functions_["v"];

    mimi::utils::PrintInfo("Creating grid functions p");
    mfem::GridFunction& p = pressure_fes.grid_functions_["p"];

    // setup v and p based on block vector
    mfem::Array<int> offsets(3);
    offsets[0] = 0;
    offsets[1] = velocity_fes.precomputed_->n_v_dofs_;
    offsets[2] = velocity_fes.precomputed_->n_v_dofs_
                 + pressure_fes.precomputed_->n_dofs_;
    block_v_p_.Update(offsets);
    v.MakeTRef(velocity_fes.fe_space_.get(), block_v_p_.GetBlock(0), 0);
    p.MakeTRef(pressure_fes.fe_space_.get(), block_v_p_.GetBlock(1), 0);

    // set initial condition
    // you can change this in python using SolutionView()
    v = 0.0;
    p = 0.0;

    // creating operators
    auto fluid_oper = std::make_unique<mimi::operators::IncompressibleFluid>(
        *velocity_fes.fe_space_,
        *pressure_fes.fe_space_);

    // let's setup the system
    // create dummy sparse matrix - mfem just sets referencee, so no copies are
    // made
    mfem::SparseMatrix tmp;
    mfem::SparseMatrix tmp_rect;

    // 1. velocity diffusion
    auto vel_diffusion =
        std::make_shared < mfem::BilinearForm(velocity_fes.fe_space_.get());
    fluid_oper->AddBilinearForm("vel_diffusion", vel_diffusion);
    assert(material_->density_ > 0.0);

    // create viscosity and density coeff for integrator
    assert(material_->viscosity_ > 0.0);
    assert(material_->density_ > 0.0);

    // temporarily disable custom parallel integrators
    // TODO: task force!
    // auto& density = Base_::coefficients_["density"];
    // density =
    // std::make_shared<mfem::ConstantCoefficient>(material_->density_); auto*
    // mass_integ = new mimi::integrators::VectorMass(density);

    // create integrator using density
    // this is owned by bilinear form, so do that first before we forget
    mfem::ConstantCoefficient mu(material_->viscosity_);
    mass->AddDomainIntegrator(new mfem::VectorDiffusion(mu));
    mass->Assemble(0);
    mass->FormSystemMatrix(velocity_fes.zero_dofs_, tmp);

    // 2. presure gradient and mass conservation
    if (material_->viscosity_ > 0.0) {
      auto conservation = std::make_shared<mfem::MixedBilinearForm>(
          velocity_fes.fe_space_.get(),
          pressure_fes.fe_space_.get());
      fluid_oper->AddMixedBilinearForm("conservation", conservation);

      // create viscosity coeff
      // auto& visc_coeff = Base_::coefficients_["viscosity"];
      // visc_coeff =
      // std::make_shared<mfem::ConstantCoefficient>(material_->viscosity_);
      // auto visc_integ = new mimi::integrators::VectorDiffusion(visc_coeff);

      // create integrator and add to mixed bilinear form
      visc->AddDomainIntegrator(new mfem::VectorDivergenceIntegrator());
      visc->Assemble(0);
      visc->FormRectangularSystemMatrix(velocity_fes.zero_dofs_,
                                        pressure_fes.zero_dofs_,
                                        tmp_rect);
    }

    // 4. linear form
    auto rhs = std::make_shared<mfem::LinearForm>(velocity_fes.fe_space_.get());
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
      rhs->SetSubVector(velocity_fes.zero_dofs_, 0.0);
      fluid_oper->AddLinearForm("rhs", rhs);
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
      Base_::linear_solvers_["incompressible_fluid"] = lin_solver;
      Base_::linear_solvers_["incompressible_fluid_preconditioner"] = prec;
    } else {
      auto lin_solver = std::make_shared<mfem::UMFPackSolver>();
      Base_::linear_solvers_["incompressible_fluid"] = lin_solver;
    }

    // setup a newton solver
    auto newton = std::make_shared<mimi::solvers::LineSearchNewton>();
    // give pointer of nl oper to control line search assembly
    newton->fluid_oper_ = fluid_oper.get();
    Base_::newton_solvers_["incompressible_fluid"] = newton;
    // basic config. you can change this using ConfigureNewton()
    newton->iterative_mode = false;
    newton->SetOperator(*fluid_oper);
    newton->SetSolver(*Base_::linear_solvers_["incompressible_fluid"]);
    newton->SetPrintLevel(mfem::IterativeSolver::PrintLevel()
                              .Warnings()
                              .Errors()
                              .Summary()
                              .FirstAndLast());
    newton->SetRelTol(1e-8);
    newton->SetAbsTol(1e-12);
    newton->SetMaxIter(MeshDim() * 10);

    // finally, register newton solver to the opeartor
    fluid_oper->SetNewtonSolver(newton);

    // TODO: timestepping
    Base_::boundary_conditions_->Print();

    // finalize operator
    fluid_oper->Setup();

    // TODO: timestepping
    // set dynamic system -> transfer ownership of those to base
    // Base_::SetDynamicSystem(fluid_oper.release(), odesolver.release());
  }

} // namespace mimi::py
