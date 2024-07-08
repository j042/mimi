#pragma once

#include <cmath>

#include "mimi/utils/ad.hpp"

namespace mimi::materials {

struct HardeningBase {
  using ADScalar_ = mimi::utils::ADScalar<double, 1>;

  virtual std::string Name() const { return "HardeningBase"; }

  virtual bool IsRateDependent() const { return false; }
  virtual bool IsTemperatureDependent() const { return false; }
  virtual void Validate() const { MIMI_FUNC() }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain) const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("HardeningBase::Evaluate not implemented");
    return {};
  }

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain,
           const double& equivalent_plastic_strain_rate) const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "HardeningBase::Evaluate (rate-dependent) not implemented");
    return {};
  }

  virtual ADScalar_ Evaluate(const ADScalar_& accumulated_plastic_strain,
                             const double& equivalent_plastic_strain_rate,
                             const double& temperature) const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "HardeningBase::Evaluate (thermo-rate-dependent) not implemented");
    return {};
  }

  virtual double SigmaY() const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError("HardeningBase::SigmaY not implemented");
    return -1.0;
  }

  virtual double
  ViscoContribution(const double equivalent_plastic_strain_rate) const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "HardeningBase::ViscoContribution not implemented");
    return {};
  }

  virtual double ThermoContribution(const double temperature) const {
    MIMI_FUNC()
    mimi::utils::PrintAndThrowError(
        "HardeningBase::ThermoContribution not implemented");
    return {};
  }
};

struct PowerLawHardening : public HardeningBase {
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

struct VoceHardening : public HardeningBase {
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

struct JohnsonCookHardening : public HardeningBase {
  using Base_ = HardeningBase;
  using ADScalar_ = Base_::ADScalar_;

  double A_;
  double B_;
  double n_;

  virtual std::string Name() const { return "JohnsonCookHardening"; }

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

struct JohnsonCookRateDependentHardening : public JohnsonCookHardening {
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
  ViscoContribution(const double equivalent_plastic_strain_rate) const {
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

  virtual ADScalar_
  Evaluate(const ADScalar_& accumulated_plastic_strain,
           const double& equivalent_plastic_strain_rate) const {
    MIMI_FUNC()

    return Base_::Evaluate(accumulated_plastic_strain)
           * ViscoContribution(equivalent_plastic_strain_rate);
  }
};

struct JohnsonCookConstantTemperatureHardening
    : public JohnsonCookRateDependentHardening {
  using Base_ = JohnsonCookRateDependentHardening;
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
  double temperature_; // this will stay constant

  // temperature contribution
  mutable double temperature_contribution_;

  /// @brief  this is a long name for a hardening model
  /// @return
  virtual std::string Name() const {
    return "JohnsonCookConstantTemperatureHardening";
  }

  virtual bool IsRateDependent() const { return true; }

  virtual bool IsTemperatureDependent() const { return false; }

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

    // compute frequently used value
    temperature_contribution_ =
        1.0
        - std::pow((temperature_ - reference_temperature_)
                       / (melting_temperature_ - reference_temperature_),
                   m_);

    mimi::utils::PrintInfo(Name(),
                           "Temp:",
                           temperature_,
                           "Ref Temp:",
                           reference_temperature_,
                           "Melting Temp:",
                           melting_temperature_,
                           "Contribution:",
                           temperature_contribution_);

    if (temperature_contribution_ <= 0.0) {
      mimi::utils::PrintAndThrowError("Invalid temperature contribution",
                                      temperature_contribution_);
    }
  }

  virtual ADScalar_ Evaluate(const ADScalar_& accumulated_plastic_strain,
                             const double& equivalent_plastic_strain_rate,
                             const double& temperature) const {
    MIMI_FUNC()

    // get visco contribution
    double visco_contribution{1.0};
    if (equivalent_plastic_strain_rate > effective_plastic_strain_rate_) {
      visco_contribution += C_
                            * std::log(equivalent_plastic_strain_rate
                                       / effective_plastic_strain_rate_);
    }

    if (std::abs(accumulated_plastic_strain.GetValue()) < 1.e-13) {
      return ADScalar_(A_) * visco_contribution * temperature_contribution_;
    }

    return (A_ + B_ * pow(accumulated_plastic_strain, n_)) * visco_contribution
           * temperature_contribution_;
  }
};

struct JohnsonCookAdiabaticRateDependentHardening
    : public JohnsonCookRateDependentHardening {
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
    return "JohnsonCookAdiabaticRateDependentHardening";
  }

  virtual bool IsRateDependent() const { return true; }

  virtual bool IsTemperatureDependent() const { return true; }

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

    // compute frequently used value
    inv_Tm_minus_Tr_ = 1. / (melting_temperature_ - reference_temperature_);
  }

  virtual ADScalar_ Evaluate(const ADScalar_& accumulated_plastic_strain,
                             const double& equivalent_plastic_strain_rate,
                             const double& temperature) const {
    MIMI_FUNC()

    return JCBase_::Evaluate(accumulated_plastic_strain)
           * ViscoContribution(equivalent_plastic_strain_rate)
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
          std::pow((temperature - reference_temperature_) * inv_Tm_minus_Tr_,
                   m_);
    }
    return thermo_contribution;
  }
};

} // namespace mimi::materials
