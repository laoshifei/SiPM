#include "SignalDependentNoise.hh"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include <vector>

bool SignalDependentNoise::Enabled(const nlohmann::json& config) {
  return config.contains("signal_dependent_noise");
}

void SignalDependentNoise::Validate(const nlohmann::json& config, const sipm::Readout& readout) {
  if (!Enabled(config)) return;
  const auto& model = config.at("signal_dependent_noise");
  if (model.at("model") != "gaussian_fir_signal_variance")
    throw std::runtime_error("Unknown signal-dependent noise model");
  const double variance = model.at("variance_per_signal_adc");
  if (!std::isfinite(variance) || variance < 0.)
    throw std::runtime_error("Signal-dependent variance must be finite and nonnegative");
  if (model.at("sample_period_ns").get<double>() != readout.sample_period_ns)
    throw std::runtime_error("Signal-dependent FIR and ADC sampling disagree");
  const auto h = model.at("coefficients").get<std::vector<double>>();
  if (h.empty() || h.size() > 1025)
    throw std::runtime_error("Invalid signal-dependent FIR length");
  double power = 0.;
  for (double x : h) {
    if (!std::isfinite(x)) throw std::runtime_error("Nonfinite signal-dependent FIR coefficient");
    power += x*x;
  }
  if (std::abs(power-1.) > 1e-8)
    throw std::runtime_error("Signal-dependent FIR must have unit squared norm");
}

void SignalDependentNoise::Apply(const nlohmann::json& config, const sipm::Readout& readout,
                                 sipm::Result& result) {
  if (!Enabled(config)) return;  // Legacy path is bit-for-bit unchanged.
  Validate(config, readout);
  const auto& model = config.at("signal_dependent_noise");
  const double variance = model.at("variance_per_signal_adc");
  if (variance == 0.) return;
  const auto h = model.at("coefficients").get<std::vector<double>>();
  const auto n = result.signal_mV.size();
  if (result.analog_mV.size() != n || result.adc_counts.size() != n)
    throw std::runtime_error("Apply signal-dependent residual after the base readout");
  const unsigned levels = 1u << readout.adc_bits;
  const double lsb = (readout.adc_max_mV-readout.adc_min_mV)/levels;
  // Distinct stream from stationary electronic noise. No transport RNG consumed.
  std::mt19937_64 engine(sipm::EventSeed(result.readout_seed,0x5349474e414cULL));
  std::normal_distribution<double> gaussian(0.,1.);
  std::vector<double> white(n+h.size()-1);
  for (double& x : white) x = gaussian(engine);
  result.n_clipped_samples = 0;
  result.adc_clipped.resize(n);
  for (std::size_t i=0; i<n; ++i) {
    double z=0.;
    for (std::size_t k=0; k<h.size(); ++k) z += h[k]*white[i+h.size()-1-k];
    // Conditional variance in ADC^2 is v * max(noiseless_signal_ADC,0).
    // Scaling after filtering makes the residual exactly zero without signal;
    // FIR lag correlations are modulated by sqrt(mu_i*mu_j). No eventwise
    // zero-mean subtraction or amplitude normalization is imposed.
    const double sigma_adc=std::sqrt(variance*std::max(result.signal_mV[i]/lsb,0.));
    const double analog=result.analog_mV[i]+lsb*sigma_adc*z;
    if (!std::isfinite(analog)) throw std::runtime_error("Nonfinite signal-dependent readout");
    const bool clipped=analog<readout.adc_min_mV || analog>=readout.adc_max_mV;
    result.analog_mV[i]=analog;
    result.adc_counts[i]=static_cast<unsigned>(std::clamp(std::floor((analog-readout.adc_min_mV)/lsb),0.,double(levels-1)));
    result.adc_clipped[i]=clipped;
    result.n_clipped_samples+=clipped;
  }
}
