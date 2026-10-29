#pragma once
#include "SiPMResponse.hh"
#include <nlohmann/json.hpp>

// Conditional two-path single-avalanche response. Both paths have unit area;
// the prompt fraction is a charge fraction, not a peak-amplitude fraction.
// This is an effective linear circuit family, not a measured S13361 circuit.
class ReadoutResponse {
public:
  static bool Enabled(const nlohmann::json& config);
  static void Validate(const nlohmann::json& config, const sipm::Config& sensor);
  static void Apply(const nlohmann::json& config, const sipm::Config& sensor,
                    double duration_ns, sipm::Result& result);
};
