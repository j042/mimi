#pragma once

#include <cmath>

#include "mimi/utils/ad.hpp"

namespace mimi::materials {

class HardeningBase {
public:
  using ADScalar_ = mimi::utils::ADScalar<double, 1>;

  HardeningBase() = default;
  virtual ~HardeningBase() {}

  virtual std::string Name() const { return "HardeningBase"; }

  virtual bool IsRateDependent() const { return false; }
  virtual bool IsTemperatureDependent() const { return false; }
  virtual void Validate() const { MIMI_FUNC() }
  virtual void InitializeTemperature(const double initial,
                                     const double melting) {
    mimi::utils::PrintDebug(
        "Doing nothing with given initial and melting temperature",
        initial,
        melting);
  }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain) const {
    mimi::utils::PrintAndThrowError("HardeningBase::Evaluate not implemented");
    return {};
  }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain,
           const double& equivalent_plastic_strain_rate) const {
    mimi::utils::PrintAndThrowError(
        "HardeningBase::Evaluate (rate-dependent) not implemented");
    return {};
  }

  virtual ADScalar_ Evaluate(const ADScalar_& accumulated_plastic_strain,
                             const double& equivalent_plastic_strain_rate,
                             const double& temperature) const {
    mimi::utils::PrintAndThrowError(
        "HardeningBase::Evaluate (thermo-rate-dependent) not implemented");
    return {};
  }

  virtual double SigmaY() const {
    mimi::utils::PrintAndThrowError("HardeningBase::SigmaY not implemented");
    return -1.0;
  }

  /// @brief Returns rate contribution given rate. By default this will
  /// return 1.
  /// @param equivalent_plastic_strain_rate
  /// @return
  virtual double
  RateContribution(const double equivalent_plastic_strain_rate) const {
    return 1.;
  }

  /// @brief Overload function with ADScalar input types. It will just extract
  /// value from it and call the function with double input
  /// @param eqps_r
  /// @return
  virtual double RateContribution(const ADScalar_& eqps_r) const {
    return RateContribution(eqps_r.GetValue());
  }

  /// Returns Thermo contribution. By default this will return 1.
  virtual double ThermoContribution(const double temperature) const {
    return 1.;
  }
};

class PowerLawHardening : public HardeningBase {
public:
  using Base_ = HardeningBase;
  using ADScalar_ = Base_::ADScalar_;

  double sigma_y_;
  double n_;
  double eps0_;

  virtual std::string Name() const { return "PowerLawHardening"; }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain) const {
    MIMI_FUNC()

    return sigma_y_ * pow(1.0 + accumulated_plastic_strain / eps0_, 1.0 / n_);
  }

  virtual double SigmaY() const { return sigma_y_; }
};

class VoceHardening : public HardeningBase {
public:
  using Base_ = HardeningBase;
  using ADScalar_ = Base_::ADScalar_;

  double sigma_y_;
  double sigma_sat_;
  double strain_constant_;

  virtual std::string Name() const { return "VoceHardening"; }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain) const {
    MIMI_FUNC()

    return sigma_sat_
           - (sigma_sat_ - sigma_y_)
                 * exp(-accumulated_plastic_strain / strain_constant_);
  }

  virtual double SigmaY() const { return sigma_y_; }
};

class JohnsonCookHardening : public HardeningBase {
public:
  using Base_ = HardeningBase;
  using ADScalar_ = Base_::ADScalar_;

  double A_;
  double B_;
  double n_;

  virtual std::string Name() const { return "JohnsonCookHardening"; }

  using Base_::Evaluate;
  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain) const {
    MIMI_FUNC()
    if (std::abs(accumulated_plastic_strain.GetValue()) < 1.e-13) {
      return ADScalar_(A_);
    }

    return A_ + B_ * pow(accumulated_plastic_strain, n_);
  }

  virtual double SigmaY() const { return A_; }
};

class JohnsonCookRateDependentHardening : public JohnsonCookHardening {
public:
  using Base_ = JohnsonCookHardening;
  using ADScalar_ = Base_::ADScalar_;

  using Base_::A_;
  using Base_::B_;
  using Base_::n_;
  double C_;
  // several ways to call this
  // Jannis calls this reference strain rate
  // wikipedia call this:
  double effective_plastic_strain_rate_;

  /// @brief  this is a long name for a hardening model
  /// @return
  virtual std::string Name() const {
    return "JohnsonCookRateDependentHardening";
  }

  virtual bool IsRateDependent() const { return true; }

  virtual double
  RateContribution(const double equivalent_plastic_strain_rate) const {
    MIMI_FUNC()

    // get visco contribution
    double visco_contribution{1.0};
    if (equivalent_plastic_strain_rate > effective_plastic_strain_rate_) {
      visco_contribution += C_
                            * std::log(equivalent_plastic_strain_rate
                                       / effective_plastic_strain_rate_);
    }
    return visco_contribution;
  }

  using Base_::Evaluate;
  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain,
           const double& equivalent_plastic_strain_rate) const {
    MIMI_FUNC()

    return Base_::Evaluate(accumulated_plastic_strain)
           * RateContribution(equivalent_plastic_strain_rate);
  }
};

class JohnsonCookTemperatureAndRateDependentHardening
    : public JohnsonCookRateDependentHardening {
public:
  using Base_ = JohnsonCookRateDependentHardening;
  using JCBase_ = Base_::Base_;
  using ADScalar_ = Base_::ADScalar_;

  using Base_::A_;
  using Base_::B_;
  using Base_::C_;
  using Base_::effective_plastic_strain_rate_;
  using Base_::n_;

  // temperature related
  double reference_temperature_;
  double melting_temperature_;
  double m_;

  // frequently used value.
  // we compute this in validate
  // TODO - restructure to be somewhere that suits better
  mutable double inv_Tm_minus_Tr_;

  /// @brief  this is a long name for a hardening model
  /// @return
  virtual std::string Name() const {
    return "JohnsonCookTemperatureAndRateDependentHardening";
  }

  virtual bool IsRateDependent() const { return true; }

  virtual bool IsTemperatureDependent() const { return true; }

  virtual void InitializeTemperature(const double initial,
                                     const double melting) {
    MIMI_FUNC()

    mimi::utils::PrintDebug("Doing nothing with given initial temperature",
                            initial);

    melting_temperature_ = melting;
  }

  virtual void Validate() const {
    MIMI_FUNC()

    if (reference_temperature_ > melting_temperature_) {
      mimi::utils::PrintAndThrowError(
          "reference temperature,",
          reference_temperature_,
          ",can't be bigger than melting temperature,",
          melting_temperature_,
          ".");
    }
  }

  using Base_::Evaluate;
  virtual ADScalar_ Evaluate(const ADScalar_& accumulated_plastic_strain,
                             const double& equivalent_plastic_strain_rate,
                             const double& temperature) const {
    MIMI_FUNC()

    return JCBase_::Evaluate(accumulated_plastic_strain)
           * RateContribution(equivalent_plastic_strain_rate)
           * ThermoContribution(temperature);
  }

  virtual double ThermoContribution(const double temperature) const {
    MIMI_FUNC()
    // then thermo
    double thermo_contribution{1.0};
    if (temperature < reference_temperature_) {
      //   nothing
    } // melted: yield stress is zero
    else if (temperature > melting_temperature_) {
      thermo_contribution = 0.0;
    } else {
      thermo_contribution -=
          std::pow((temperature - reference_temperature_)
                       / (melting_temperature_ - reference_temperature_),
                   m_);
    }
    return thermo_contribution;
  }
};

struct JohnsonCookConstantTemperatureHardening
    : public JohnsonCookTemperatureAndRateDependentHardening {
  using Base_ = JohnsonCookTemperatureAndRateDependentHardening;
  using ADScalar_ = Base_::ADScalar_;

  double temperature_{-1.0};
  double temperature_contribution_{-1.0};

  /// @brief  this is a long name for a hardening model
  /// @return
  virtual std::string Name() const {
    return "JohnsonCookConstantTemperatureHardening";
  }

  virtual bool IsRateDependent() const { return true; }

  /// Considered temperature independent to minimize unnecessary computations
  virtual bool IsTemperatureDependent() const { return false; }

  virtual void InitializeTemperature(const double initial,
                                     const double melting) {
    MIMI_FUNC()

    melting_temperature_ = melting;
    SetTemperature(initial);
  }

  void SetTemperature(const double temp) {
    MIMI_FUNC()
    temperature_ = temp;
    temperature_contribution_ =
        1.0
        - std::pow((temperature_ - reference_temperature_)
                       / (melting_temperature_ - reference_temperature_),
                   m_);

    if (temperature_contribution_ <= 0.0) {
      mimi::utils::PrintAndThrowError("Invalid temperature contribution",
                                      temperature_contribution_);
    }
  }

  double GetTemperature() const { return temperature_; }

  /// As for constant temperature, input is ignored. In debug mode, checks if
  /// they match
  virtual double ThermoContribution(const double temperature) const {
    MIMI_FUNC()

    assert(std::abs(temperature_ - temperature) < 1.e-12);

    return temperature_contribution_;
  }

  using Base_::Evaluate;
  /// Will raise as this isn't supposed to be temperature dependent
  virtual ADScalar_ Evaluate(const ADScalar_& accumulated_plastic_strain,
                             const double& equivalent_plastic_strain_rate,
                             const double& temperature) const {

    mimi::utils::PrintAndThrowError(Name(),
                                    "cannot have temperature as input.");

    return {};
  }
};

} // namespace mimi::materials
