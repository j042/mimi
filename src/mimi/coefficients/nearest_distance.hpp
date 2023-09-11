#pragma once

#include <memory>
#include <vector>

#include <splinepy/py/py_spline.hpp>

#include "mimi/utils/containers.hpp"

namespace mimi::coefficients {

class NearestDistanceBase {
public:
  double coefficient_{1e-4};
  double tolerance_{1e-12};
  int para_dim_{-1};
  int dim_{-1};

  struct Query {
    mimi::utils::Data<double> query_;
    mimi::utils::Data<double> initial_guess_;
    int max_iterations_;
  };

  struct Results {
    // query
    mimi::utils::Data<double> parametric_;
    mimi::utils::Data<double> physical_;
    mimi::utils::Data<double> physical_minus_query_;
    double distance_{};
    double convergence_{};
    mimi::utils::Data<double, 2> first_derivatives_;
    mimi::utils::Data<double, 3> second_derivatives_;

    // locally save dim infos
    int para_dim_{};
    int dim_{};

    // we save some information about the query result
    // so that we can avoid recomputing these values for J(R(u))
    // it makes the most sense as query and result pair, but
    // we retrieve results with ids, so we do it this way.
    bool active_{};
    double query_metric_tensor_weight_{};

    /// given spline's para_dim and dim, allocates result's size.
    /// set give biggest acceptable size.
    void SetSize(const int& para_dim, const int& dim) {
      MIMI_FUNC()

      para_dim_ = para_dim;
      dim_ = dim;

      parametric_.SetSize(para_dim);
      physical_.SetSize(dim);
      physical_minus_query_.SetSize(dim);
      first_derivatives_.SetSize(para_dim * dim);
      first_derivatives_.stride0_ = dim;
      second_derivatives_.SetSize(para_dim * para_dim * dim);
      second_derivatives_.stride0_ = para_dim * dim;
      second_derivatives_.stride1_ = dim;
    }
  };

  NearestDistanceBase() = default;

  virtual ~NearestDistanceBase() = default;
  virtual void NearestDistance(const Query& query, Results& results) const {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("Inherit and implement me");
  };

  /// Related body count
  virtual int Size() const {
    MIMI_FUNC()

    mimi::utils::PrintAndThrowError("Inherit and implement me.");

    return -1;
  };
};

class NearestDistanceToSplines : public NearestDistanceBase {
protected:
  std::vector<std::shared_ptr<splinepy::py::PySpline>> splines_;

public:
  void Clear() {
    MIMI_FUNC()

    splines_.clear();
  }

  void AddSpline(std::shared_ptr<splinepy::py::PySpline>& spline) {
    MIMI_FUNC()

    // technically, you can use any combination of splines as long as
    // physical dimension matches.
    // however, for implementation purposes, we will only accept boundary
    // splines
    const int para_dim = spline->Core()->SplinepyParaDim();
    const int dim = spline->Core()->SplinepyDim();

    // set this scene's dimension
    if (splines_.size() < 0) {
      para_dim_ = para_dim;
      dim_ = dim;
    }

    splines_.push_back(spline);
  }

  void PlantKdTree(const int resolution, const int nthreads) {
    MIMI_FUNC()
    assert(splines_.size());

    for (auto& spline : splines_) {
      const int para_dim = spline->Core()->SplinepyParaDim();
      mimi::utils::Data<int> res_query(para_dim);
      res_query.fill(resolution);
      spline->Core()->SplinepyPlantNewKdTreeForProximity(res_query.data(),
                                                         nthreads);
    }
  }

  /// TODO currently really will return last spline. this need to change
  virtual void NearestDistance(const Query& query, Results& results) const {
    MIMI_FUNC()

    /// temp sanity check
    assert(splines_.size() == 1);

    for (const auto& spline : splines_) {
      spline->Core()->SplinepyVerboseProximity(
          query.query_.data(),
          tolerance_,
          query.max_iterations_,
          false, /* aggressive_bounds */
          results.parametric_.data(),
          results.physical_.data(),
          results.physical_minus_query_.data(),
          results.distance_,
          results.convergence_,
          results.first_derivatives_.data(),
          results.second_derivatives_.data());
    }
  }

  virtual int Size() const {
    MIMI_FUNC()

    return splines_.size();
  }
};

} // namespace mimi::coefficients
