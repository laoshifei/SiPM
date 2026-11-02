#pragma once
#include <cstdint>
#include <vector>
#include <nlohmann/json.hpp>

// Conditional Gaussian sum/difference components with separate variance state.
// Their sum is the legacy innovation and their unconditional spectrum sums to
// unity. This empirical state is not a physical SiPM gain fluctuation.
class SpectralVarianceSplit {
public:
  static void Validate(const nlohmann::json& signal_noise);
  static std::vector<double> HighInnovations(const std::vector<double>& white,
                                           std::uint64_t seed, unsigned delay);
  static void Apply(const nlohmann::json& signal_noise, std::uint64_t seed,
                    std::vector<double>& white);
};
