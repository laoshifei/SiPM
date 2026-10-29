#include "ReadoutNoise.hh"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

bool ReadoutNoise::Enabled(const nlohmann::json& config) {
  return config.contains("electronics_noise");
}

void ReadoutNoise::Validate(const nlohmann::json& config) {
  if (!Enabled(config)) return;
  const auto& noise = config.at("electronics_noise");
  if (noise.at("model") != "gaussian_fir") throw std::runtime_error("Unknown electronics noise model");
  const auto coefficients = noise.at("coefficients").get<std::vector<double>>();
  if (coefficients.empty() || coefficients.size() > 1025) throw std::runtime_error("Invalid noise FIR length");
  double power = 0.;
  for (double x : coefficients) {
    if (!std::isfinite(x)) throw std::runtime_error("Nonfinite noise FIR coefficient");
    power += x*x;
  }
  if (std::abs(power-1.) > 1e-8) throw std::runtime_error("Noise FIR must have unit squared norm");
}

void ReadoutNoise::Apply(const nlohmann::json& config, const sipm::Readout& readout,
                         sipm::Result& result) {
  if (!Enabled(config)) return;
  Validate(config);
  const auto coefficients = config.at("electronics_noise").at("coefficients").get<std::vector<double>>();
  const auto n = result.signal_mV.size();
  // Random prehistory gives stationary variance from the first ADC sample.
  std::mt19937_64 engine(result.readout_seed);
  std::normal_distribution<double> gaussian(0., 1.);
  std::vector<double> white(n+coefficients.size()-1);
  for (double& x : white) x = gaussian(engine);
  result.analog_mV.resize(n); result.adc_counts.resize(n); result.adc_clipped.resize(n);
  result.n_clipped_samples = 0;
  const unsigned levels = 1u << readout.adc_bits;
  const double lsb = (readout.adc_max_mV-readout.adc_min_mV)/levels;
  for (std::size_t i = 0; i < n; ++i) {
    double noise = 0.;
    for (std::size_t k = 0; k < coefficients.size(); ++k) noise += coefficients[k]*white[i+coefficients.size()-1-k];
    const double analog = result.signal_mV[i]+readout.baseline_mV+readout.noise_rms_mV*noise;
    const bool clipped = analog < readout.adc_min_mV || analog >= readout.adc_max_mV;
    result.analog_mV[i] = analog;
    result.adc_counts[i] = static_cast<unsigned>(std::clamp(std::floor((analog-readout.adc_min_mV)/lsb),0.,double(levels-1)));
    result.adc_clipped[i] = clipped; result.n_clipped_samples += clipped;
  }
}
