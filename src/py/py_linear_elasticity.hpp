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
  double mu_, lambda_, rho_;
  using Base_ = PySolid;

  PyLinearElasticity() = default;

  virtual void
  SetParameters(const double mu, const double lambda, const double rho) {
    mu_ = mu;
    lambda_ = lambda;
    rho_ = rho;
  }

  virtual void
  SetParameters_(const double young, const double poisson, const double rho) {
    auto const& [mu, lambda] = mimi::utils::ToLambdaAndMu(young, poisson);
    mu_ = mu, lambda_ = lambda;
    rho_ = rho;
  }

  virtual void Setup() {
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
    // 1. mass
    auto mass = std::make_shared<mfem::BilinearForm>(disp_fes.fe_space_.get());
    auto mass_integ = std::make_unique<mimi::integrators::VectorMass> mass
                          ->AddDomainIntegrator()
  }
};

} // namespace mimi::py
