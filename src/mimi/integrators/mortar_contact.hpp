#pragma once

#include <cmath>

#include <mfem.hpp>

#include <splinepy/py/py_spline.hpp>

#include "mimi/coefficients/nearest_distance.hpp"
#include "mimi/integrators/integrator_utils.hpp"
#include "mimi/integrators/nonlinear_base.hpp"
#include "mimi/utils/containers.hpp"
#include "mimi/utils/precomputed.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::integrators {

/// implements normal contact methods presented in "De Lorenzis.
/// A large deformation frictional contact formulation using NURBS-based
/// isogeometric analysis"
/// Currently implements normal contact
class MortarContact : public NonlinearBase {
public:
  using Base_ = NonlinearBase;
  template<typename T>
  using Vector_ = mimi::utils::Vector<T>;
  using ElementQuadData_ = mimi::utils::ElementQuadData;
  using ElementData_ = mimi::utils::ElementData;
  using QuadData_ = mimi::utils::QuadData;

  // post process
  double last_area_; // for p = forces / A
  double last_pressure_;
  mimi::utils::Data<double> last_force_;

protected:
  /// scene
  std::shared_ptr<mimi::coefficients::NearestDistanceBase>
      nearest_distance_coeff_ = nullptr;

  /// convenient constants - space dim (dim_) is in base
  int dim_;
  int boundary_para_dim_;
  int n_threads_;
  int n_marked_boundaries_;

  /// residual contribution indices
  /// size == n_marked_boundaries_;
  Vector_<int> marked_boundary_v_dofs_;
  Vector_<int> local_marked_v_dofs_;
  Vector_<int> local_marked_dofs_;

  /// two vectors for load balancing
  /// first we visit all quad points and also mark quad activity
  Vector_<bool> element_activity_;
  Vector_<int> active_elements_;

  /// these are numerator and denominator
  mfem::Vector average_gap_;
  mfem::Vector average_pressure_;
  mfem::Vector area_;

  Vector_<MortarContactWorkData> work_data_;

public:
  MortarContact(
      const std::shared_ptr<mimi::coefficients::NearestDistanceBase>&
          nearest_distance_coeff,
      const std::string& name,
      const std::shared_ptr<mimi::utils::PrecomputedData>& precomputed)
      : NonlinearBase(name, precomputed),
        nearest_distance_coeff_(nearest_distance_coeff) {}

  /// Prepares spaces for each thread and computes index mappings for local
  /// pressure array for marked boundaries
  virtual void Prepare();

  /// Sets gap and pressure to zero
  void InitializeGapAreaPressure();

  /// Element assembly for g and A
  void ElementGapAndArea(const Vector_<QuadData_>& q_data,
                         MortarContactWorkData& w,
                         double& area_contribution) const;

  /// Computes average gap. This loops over the whole boundary as in our
  /// application, we need to compute current area and it's done here.
  void ComputePressure(const mfem::Vector& current_u, double& area);

  /// assembles contact traction with pressure. make sure pressure is non zero
  /// (IsPressureZero), otherwise this will be a waste
  template<bool record>
  void ElementResidual(const Vector_<QuadData_>& q_data,
                       MortarContactWorkData& w,
                       mimi::utils::Data<double>& force /* only if applicable*/,
                       double& pressure_integral) const {
    MIMI_FUNC()
    w.ResidualMatrix() = 0.0;
    const double penalty = nearest_distance_coeff_->coefficient_;
    double* residual_d = w.ResidualMatrix().GetData();

    for (const QuadData_& q : q_data) {
      const double p = w.Pressure(q.N);
      assert(std::isfinite(p));
      assert(p < 0.);

      // we need derivatives to get normal.
      mfem::DenseMatrix& J = w.ComputeJ(q.dN_dxi);
      const double det_J = J.Weight();
      ComputeUnitNormal(J, w.normal_);

      // this one has negative sign as we use the normal from gauss point
      const double fac = q.integration_weight * det_J * p;
      Ptr_AddMult_a_VWt(-fac,
                        q.N.begin(),
                        q.N.end(),
                        w.normal_.begin(),
                        w.normal_.end(),
                        residual_d);

      // this is application specific quantity
      if constexpr (record) {
        force.Add(fac, w.normal_.begin());
        pressure_integral += fac;
      }
    }
  }

  /// Same rule applies as ElementResidual - check if pressure is non zero.
  /// Note - we can probably generalize this with fold expressions for all
  /// integrators
  void ElementResidualAndGrad(
      const Vector_<QuadData_>& q_data,
      MortarContactWorkData& w,
      mimi::utils::Data<double>& force /* only if applicable*/,
      double& pressure_integral);

  /// rhs (for us, we have them on lhs - thus fliped sign) - used in line
  /// search
  virtual void AddBoundaryResidual(const mfem::Vector& current_u,
                                   mfem::Vector& residual);

  /// Currently unused interface. Use AddBoundaryResidualAndGrad()
  virtual void AddBoundaryGrad(const mfem::Vector& current_u,
                               mfem::SparseMatrix& grad) const {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError(
        "Currently not implemented, use AddDomainResidualAndGrad");
  }

  /// assembles residual and grad at the same time. this should also have
  /// assembled the last converged residual
  virtual void AddBoundaryResidualAndGrad(const mfem::Vector& current_u,
                                          const double grad_factor,
                                          mfem::Vector& residual,
                                          mfem::SparseMatrix& grad);

  /// checks GapNorm from given test_u. Useful to check fulfillment of contact
  /// conditions.
  virtual double GapNorm(const mfem::Vector& test_u, const int nthreads);

  /// Saves relevant properties at converged state
  virtual void BoundaryPostTimeAdvance(const mfem::Vector& converged_u);
};

} // namespace mimi::integrators
