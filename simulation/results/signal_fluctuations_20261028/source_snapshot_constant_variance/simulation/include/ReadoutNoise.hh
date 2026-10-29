#pragma once
#include "SiPMResponse.hh"
#include <nlohmann/json.hpp>

// Stationary electronic noise only. It never changes photons or avalanche charge.
class ReadoutNoise {
public:
  static bool Enabled(const nlohmann::json& config);
  static void Validate(const nlohmann::json& config);
  static void Apply(const nlohmann::json& config, const sipm::Readout& readout,
                    sipm::Result& result);
};
