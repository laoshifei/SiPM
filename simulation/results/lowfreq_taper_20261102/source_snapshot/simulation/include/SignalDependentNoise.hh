#pragma once
#include "SiPMResponse.hh"
#include <nlohmann/json.hpp>

// Optional empirical covariance residual at the ADC input. This is calibrated
// per acquisition, not from isotope identity or experimental particle labels.
// It leaves transport, avalanches and the deterministic signal unchanged.
class SignalDependentNoise {
public:
  static bool Enabled(const nlohmann::json&);
  static void Validate(const nlohmann::json&, const sipm::Readout&);
  static void Apply(const nlohmann::json&, const sipm::Readout&, sipm::Result&);
};
