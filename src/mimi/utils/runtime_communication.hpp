#pragma once

#include <unordered_map>
#include <unordered_set>

#include <mfem.hpp>

#include <cnpy.h>

#include "mimi/utils/containers.hpp"
#include "mimi/utils/print.hpp"

namespace mimi::utils {

template<typename MapType>
typename MapType::mapped_type
MapDefaultGet(const MapType& map,
              const typename MapType::key_type& key,
              const typename MapType::mapped_type& default_) {
  auto val = map.find(key);
  if (val == map.end()) {
    // maybe add quiet option
    mimi::utils::PrintInfo("RuntimeCommunication -",
                           key,
                           "requested, but it's not saved. Returning default.");
    return default_;
  }
  return val->second;
}

template<typename MapType>
void MapSet(MapType& map,
            const typename MapType::key_type& key,
            const typename MapType::mapped_type& value) {
  auto val = map.find(key);
  if (val != map.end()) {
    mimi::utils::PrintInfo("RuntimeCommunication -",
                           key,
                           "exists, overwriting.");
    val->second = value;
    return;
  }
  map[key] = value;
}

/// @brief A visitor object that carries runtime configured data and exchanges
/// data. Should goto all the nonlinear integrators.
class RuntimeCommunication {
protected:
  std::unordered_map<std::string, Vector<mfem::Vector>> vectors_;
  std::unordered_map<std::string, Vector<double>> latest_vector_;
  std::unordered_map<std::string, Vector<double>> real_history_;
  std::unordered_map<std::string, Vector<int>> integer_history_;
  std::unordered_map<std::string, double> real_;
  std::unordered_map<std::string, int> integer_;
  std::unordered_map<std::string, int> save_those_vectors_every_;

  int i_timestep_;
  double t_;

  const std::string& FName() const {
    if (fname_.size() == 0) {
      mimi::utils::PrintInfo(
          "Save requested, but fname not set in RuntimeCommunication");
    }
    return fname_;
  }

public:
  std::string fname_{};

  RuntimeCommunication() = default;

  void InitializeTimeStep() {
    MIMI_FUNC()
    i_timestep_ = 0;
    t_ = 0.0;
  }

  void NextTimeStep(const double dt) {
    ++i_timestep_;
    t_ += dt;
  }

  void SetFName(const std::string& fname) {
    MIMI_FUNC()

    fname_ = fname;
  }

  double GetReal(const std::string& key, const double default_) const {
    MIMI_FUNC()

    return MapDefaultGet(real_, key, default_);
  }

  void SetReal(const std::string& key, const double value) {
    MIMI_FUNC()

    MapSet(real_, key, value);
  }

  int GetInteger(const std::string& key, const int default_) const {
    MIMI_FUNC()

    return MapDefaultGet(integer_, key, default_);
  }

  void SetInteger(const std::string& key, const int value) {
    MIMI_FUNC()

    MapSet(integer_, key, value);
  }

  void AppendShouldSave(const std::string& name, const int every) {
    MIMI_FUNC()

    save_those_vectors_every_[name] = every;
  }

  bool ShouldSave(const std::string& name) const {
    MIMI_FUNC()

    auto search = save_those_vectors_every_.find(name);
    if (search == save_those_vectors_every_.end())
      return false;

    // value contains "every"
    return i_timestep_ % search->second == 0;
  }

  void SetupRealHistory(const std::string& name, const int n_reserve) {
    MIMI_FUNC()

    real_history_[name].reserve(n_reserve);
  }

  void RecordRealHistory(const std::string& name, const double value) {
    MIMI_FUNC()

    real_history_.at(name).push_back(value);
  }

  Vector<double>& GetRealHistory(const std::string& name) {
    MIMI_FUNC()

    return real_history_.at(name);
  }

  void SaveRealHistory(const std::string& name) {
    MIMI_FUNC()

    Vector<double>& history = GetRealHistory(name);

    SaveVector(name + "_history", history.data(), history.size());
  }

  template<typename T, typename IndexType>
  void SaveVector(const std::string& vector_name,
                  const T* buf,
                  const IndexType buf_size) const {
    MIMI_FUNC()

    cnpy::npz_save(FName(),
                   vector_name,
                   buf,
                   {static_cast<size_t>(buf_size)},
                   "a");
  }

  void SaveVector(const std::string& vector_name,
                  const mfem::Vector& vector) const {
    MIMI_FUNC()

    SaveVector(vector_name, vector.GetData(), vector.Size());
  }

  void SaveDynamicVector(const std::string& vector_name,
                         const mfem::Vector& vector) {
    MIMI_FUNC()

    SaveVector(vector_name + std::to_string(i_timestep_),
               vector.GetData(),
               vector.Size());
    auto& vec = latest_vector_[vector_name];
    vec.resize(vector.Size());
    std::copy_n(vector.GetData(), vector.Size(), vec.data());
  }

  Vector<double>& LatestVector(const std::string& vector_name) {
    return latest_vector_.at(vector_name);
  }
};

} // namespace mimi::utils
