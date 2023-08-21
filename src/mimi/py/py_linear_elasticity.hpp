#include <memory>
#include <string>
#include <unordered_map>

/* pybind11 */
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// splinepy
#include <splinepy/py/py_spline.hpp>

// mimi
#include "mimi/py/py_solid.hpp"

namespace mimi::py {

class PyLinearElasticity : PySolid {
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

    // create displacement
    auto& disp_fes = Base_::fe_spaces_["displacement"];
    disp_fes.fe_space_ =
        std::make_unique<mfem::FiniteElementSpace>(Base_::Mesh().get(),
                                                   Base_::Mesh()->NURBSext,
                                                   fe_collection,
                                                   MeshDim());

    // create solution fieds for displacements
    mfem::GridFunction& x = disp_fes.grid_functions_["x"];
    x.SetSpace(disp_fes.fe_space_.get());

    // and velocity
    mfem::GridFunctio& v = disp_fes.grid_functions_["v"];
    v.SetSpace(disp_fes.fe_space_.get());

    // and reference / initial reference. initialize with fe space then
    // copy from mesh
    mfem::GridFunction& x_ref = disp_fes.grid_functions_["x_ref"];
    x_ref.SetSpace(disp_fes.fe_space_.get());
    Base_::Mesh()->GetNodes(x_ref);

    // set initial condition
    x = x_ref;
    v = 0.0;

    /// find bdr dof ids
    Base_::FindBoundaryDofIds();

    // create a linear elasticity operator
    // at first, this is empty
    auto le_oper = std::make_unique<mimi::operators::LinearElasticity>(
        *disp_fes.fe_space_);

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
    auto* mass_integ = new mimi::integrators::VectorMass(*rho);
    mass->AddDomainIntegrator(mass_integ);

    // precompute using nthread
    mass_integ->ComputeElementMatrices(*disp_fes.fe_space_, nthreads);

    // assemble and remove zero bc entries
    mass->Assemble();
    mass->FromSystemMatrix(disp_fes.zero_dofs_, tmp);

    // release some memory
    mass_integ.element_matrices_.reset(); // release saved matrices

    // 2. viscosity
    if (viscosity > 0.0) {
      auto visc =
          std::make_shared<mfem::BilinearForm>(disp_fes.fe_space_.get());
      le_oper->AddBilinearForm("viscosity", visc);

      // create viscosity coeff
      auto& visc_coeff = Base_::coefficients_["viscosity"];
      visc_coeff = std::make_shared<mfem::ConstantCoefficient>(viscosity_);

      // create integrator using viscosity
      // and add it to bilinear form
      auto visc_integ = new mimi::integrators::VectorDiffusion(*visc_coeff);
      visc->AddDomainIntegrator(visc_integ);

      // nthread assemble
      visc_integ->ComputeElemeentMatrices(*disp_fes.fe_space_, nthreads);

      visc->Assemble();
      visc->FormSystemMatrix(disp_fes.zero_dofs_, tmp);

      visc_integ.element_matrices_.reset();
    }

    // 3. Lin elasitiy stiffness
    // start with bilin form
    auto stiffness =
        std::make_shared<mfem::BilinearForm>(disp_fec.fe_space_.get());
    le_oper->AddBilinearForm("stiffness", stiffness);

    // create coeffs
    auto& mu = Base_::coefficients_["mu"];
    mu = std::make_shared<mfem::ConstantCoefficient>(mu_);
    auto& lambda = Base_::coefficients_["lambda"];
    lambda = std::make_shared<mfem::ConstantCoefficient>(lambda_);

    // create integ
    auto stiffness_integ =
        new mimi::integrators::LinearElasticity(*lambda, *mu);
    stifness->AddDomainIntegrator(stiffness_integ);

    // nthread assemble
    stiffness_integ->ComputeElementMatrices(*disp_fes.fe_space_, nthreads);

    stiffness->Assemble();
    stiffness->FormSystemMatrix(disp_fes.zero_dofs_, tmp);

    stiffness_integ.element_matrices_.reset();

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
      rhs->AddDomainIntegrator(new mfem::VectorDomainLFIntegrator(b_force_coeff));
    }

    if (const auto& traction = Base_::boundary_conditions_->InitialConfiguration().traction_;
        traction.size() != 0) {

        rhs_set = true;

        // create coeff
        auto traction_coeff = std::make_shared<mfem::VectorArrayCoefficient>(Base_::MeshDim());
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
            auto tpd = traction_per_dim[dim];
            tpd(bid) = value;
          }
        }

        // set
        for (int i{}; i < traction_per_dim.size(); ++i) {
          traction_coeff.Set(i, traction_per_dim[i]);
        }

        // add to linear form
        rhs->AddBoundaryIntegrator(new mfem::VectorBoundaryLFIntegrator(traction_coeff));
    }

    if (rhs_set) {
      rhs->Assemble();
      Base_::AddLinearForm("rhs", rhs);
    }


    // ode
    auto gen_alpha = std::make_unique<mimi::solvers::GeneralizedAlpha2>(*le_oper);
    gen_alpha->PrintInfo();

    // set dynamic system -> transfer ownership
    Base_::SetDynamicSystem2(le_oper.release(), gen_alpha.release());
  }
};

} // namespace mimi::py
