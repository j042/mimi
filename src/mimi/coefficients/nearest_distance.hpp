#pragma once

#include <memory>
#include <vector>

#include <splinepy/py/py_spline.hpp>

#include "mimi/utils/containers.hpp"

namespace mimi::coefficients {

/// @brief base class for nearest distance coefficients. Helps find nearest
/// distance of foreign objects and contact formulations. Some contact related
/// functions are implemented here.
class NearestDistanceBase {
public:
  /// @brief coefficient used for contact formulation
  double coefficient_{1e4};
  /// @brief tolerance for nearest search
  double tolerance_{1e-24};
  /// @brief parametric dimension of the scene
  int para_dim_{-1};
  /// @brief physical dimension of the scene
  int dim_{-1};

  /// @brief nearest distance query
  struct Query {
    mimi::utils::Data<double> query_;
    // mimi::utils::Data<double> initial_guess_;
    int max_iterations_ = -1;

    void SetSize(const int& dim) {
      query_.Reallocate(dim);
      // initial_guess_.Reallocate(para_dim);
    }

    template<typename Stream>
    void Print(Stream& out) const {
      out << "Query: [";
      for (const auto& q : query_) {
        out << q << ", ";
      }
      out << "]\n";
    }
  };

  /// @brief results of nearest distance query
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
    // bool active_{};
    // double query_metric_tensor_weight_{};
    mimi::utils::Data<double> normal_;
    double normal_norm_;

    template<typename Stream>
    void Print(Stream& out) const {
      out << "Parametric Coordinate: [";
      for (const auto& q : parametric_) {
        out << q << ", ";
      }
      out << "]\n";
      out << "Physical Coordinate: [";
      for (const auto& q : physical_) {
        out << q << ", ";
      }
      out << "]\n";
      out << "Physical Coordinate - Query: [";
      for (const auto& q : physical_minus_query_) {
        out << q << ", ";
      }
      out << "]\n";
      out << "Distance: " << distance_ << "\n";
      out << "Convergence: " << convergence_ << "\n";
      out << "First Derivatives: [";
      for (const auto& q : first_derivatives_) {
        out << q << ", ";
      }
      out << "]\n";
      out << "Second derivatives: [";
      for (const auto& q : second_derivatives_) {
        out << q << ", ";
      }
      out << "]\n";
      out << "Normal: [";
      for (const auto& q : normal_) {
        out << q << ", ";
      }
      out << "]\n";
      out << "Normal norm: " << normal_norm_ << "\n";
    }

    /// given spline's para_dim and dim, allocates result's size.
    /// set give biggest acceptable size.
    void SetSize(const int& para_dim, const int& dim) {
      MIMI_FUNC()

      para_dim_ = para_dim;
      dim_ = dim;

      if (dim_ > 3 || dim_ < 2) {
        mimi::utils::PrintAndThrowError("Unsupported Dim:", dim_);
      }
      if (para_dim + 1 != dim) {
        mimi::utils::PrintAndThrowError(
            "boundary para_dim should be one smaller than dim.");
      }

      parametric_.Reallocate(para_dim);
      physical_.Reallocate(dim);
      physical_minus_query_.Reallocate(dim);
      first_derivatives_.Reallocate(para_dim * dim);
      first_derivatives_.SetShape(para_dim, dim);
      second_derivatives_.Reallocate(para_dim * para_dim * dim);
      second_derivatives_.SetShape(para_dim, para_dim, dim);
      normal_.Reallocate(dim);
    }

    /// normal - we define a rule here, and it'd be your job to prepare
    /// the foreign spline in a correct orientation
    /// Note that this operation is not thread-safe, since each thread is
    /// expected to have its own copy of this object.
    template<bool unit_normal = true>
    inline mimi::utils::Data<double>& ComputeNormal() {
      MIMI_FUNC()

      if (dim_ == 2) {
        const double& d0 = first_derivatives_[0];
        const double& d1 = first_derivatives_[1];

        if constexpr (unit_normal) {
          normal_norm_ = std::sqrt(d0 * d0 + d1 * d1);
          const double inv_norm2 = 1. / normal_norm_;
          normal_[0] = d1 * inv_norm2;
          normal_[1] = -d0 * inv_norm2;
        } else {
          normal_[0] = d1;
          normal_[1] = -d0;
        }

        // this should be either 2d or 3d
      } else {
        const double& d0 = first_derivatives_[0];
        const double& d1 = first_derivatives_[1];
        const double& d2 = first_derivatives_[2];
        const double& d3 = first_derivatives_[3];
        const double& d4 = first_derivatives_[4];
        const double& d5 = first_derivatives_[5];

        if constexpr (unit_normal) {
          const double n0 = d1 * d5 - d2 * d4;
          const double n1 = d2 * d3 - d0 * d5;
          const double n2 = d0 * d4 - d1 * d3;

          const double inv_norm2 = 1. / std::sqrt(n0 * n0 + n1 * n1 + n2 * n2);

          normal_[0] = n0 * inv_norm2;
          normal_[1] = n1 * inv_norm2;
          normal_[2] = n2 * inv_norm2;

        } else {
          normal_[0] = d1 * d5 - d2 * d4;
          normal_[1] = d2 * d3 - d0 * d5;
          normal_[2] = d0 * d4 - d1 * d3;
        }
      }
      return normal_;
    }

    // this will fill normal at the same time
    inline double NormalGap() const {
      MIMI_FUNC()

      // here, we apply negative sign to physical_minus_query
      // normal gap is formulated as query minus physical
      return -normal_.InnerProduct(physical_minus_query_);
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
  }
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
      res_query.Fill(resolution);
      spline->Core()->SplinepyPlantNewKdTreeForProximity(res_query.data(),
                                                         nthreads);
    }
  }

  /// TODO currently really will return last spline. this need to change
  virtual void NearestDistance(const Query& query, Results& results) const {
    MIMI_FUNC()

    /// temp sanity check
    assert(splines_.size() == 1);

    // we should keep a temporary result
    // and copy the minimum distance results
    for (const auto& spline : splines_) {
      spline->Core()->SplinepyVerboseProximity(
          query.query_.data(),
          tolerance_,
          query.max_iterations_,
          true, /* aggressive_bounds */
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
