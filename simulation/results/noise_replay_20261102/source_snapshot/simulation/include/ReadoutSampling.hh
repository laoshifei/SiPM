#pragma once
#include "SiPMResponse.hh"
#include <nlohmann/json.hpp>

// Diagnostic aperture offsets on a fixed native avalanche history. Phase zero
// follows the production path exactly; other phases use native fine-bin charge
// integration, followed by fresh coarse-sample electronics/residual noise.
class ReadoutSampling {
public:
  static void Build(const nlohmann::json&, const sipm::Config&, double duration_ns,
                    double phase_ns, sipm::Result&);
};
