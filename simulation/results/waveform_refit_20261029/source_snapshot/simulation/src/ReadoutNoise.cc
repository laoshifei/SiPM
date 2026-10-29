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
  const double cv = noise.value("event_variance_cv", 0.);
  const double baseline_sigma = noise.value("baseline_jitter_mV", 0.);
  const double correlation = noise.value("baseline_logvariance_correlation", 0.);
  if (!std::isfinite(cv) || cv < 0. || cv > 2.
      || !std::isfinite(baseline_sigma) || baseline_sigma < 0.
      || !std::isfinite(correlation) || std::abs(correlation) > .99
      || (correlation != 0. && baseline_sigma == 0.))
    throw std::runtime_error("Invalid event electronics-noise state parameters");
  auto validate_state = [](const nlohmann::json& state) {
    const double m=state.at("baseline_mV"), s=state.at("noise_rms_mV");
    const double b=state.at("baseline_jitter_mV"), v=state.at("event_variance_cv");
    const double rho=state.value("baseline_logvariance_correlation",0.);
    if (!std::isfinite(m) || !std::isfinite(s) || s<0. || !std::isfinite(b) || b<0.
        || !std::isfinite(v) || v<0. || v>2. || !std::isfinite(rho) || std::abs(rho)>.99
        || (rho!=0. && b==0.))
      throw std::runtime_error("Invalid baseline mixture state");
  };
  if (noise.contains("states")) {
    const auto& states=noise.at("states");
    if (!states.is_array() || states.empty() || states.size()>16)
      throw std::runtime_error("Invalid baseline-state mixture size");
    double total=0.;
    for (const auto& state:states) {
      validate_state(state);
      const double weight=state.at("weight");
      if (!std::isfinite(weight) || weight<0.) throw std::runtime_error("Invalid baseline-state weight");
      total+=weight;
    }
    if (std::abs(total-1.)>1e-8) throw std::runtime_error("Baseline-state weights must sum to one");
  }
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
  // Optional acquisition-state scatter. Independent streams preserve the
  // legacy FIR innovations and keep transport/avalanche random numbers intact.
  const auto& model = config.at("electronics_noise");
  const nlohmann::json* selected=&model;
  double noise_sigma = readout.noise_rms_mV;
  double baseline = readout.baseline_mV;
  if (model.contains("states")) {
    std::mt19937_64 selector(sipm::EventSeed(result.readout_seed,0x4241534553ULL));
    const double u=std::uniform_real_distribution<double>(0.,1.)(selector);
    double cumulative=0.;
    const auto& states=model.at("states");selected=&states.back();
    for (const auto& state:states) {cumulative+=state.at("weight").get<double>();if (u<cumulative) {selected=&state;break;}}
    noise_sigma=selected->at("noise_rms_mV");baseline=selected->at("baseline_mV");
  }
  const double baseline_sigma = selected->value("baseline_jitter_mV", 0.);
  double zbaseline=0.;
  if (baseline_sigma>0.) {
    std::mt19937_64 state(sipm::EventSeed(result.readout_seed,0x424153454dULL));
    std::normal_distribution<double> normal(0.,1.);zbaseline=normal(state);
    baseline+=baseline_sigma*zbaseline;
  }
  const double cv = selected->value("event_variance_cv", 0.);
  if (cv > 0.) {
    std::mt19937_64 state(sipm::EventSeed(result.readout_seed, 0x4241534556ULL));
    std::normal_distribution<double> normal(0., 1.);
    const double sigma2 = std::log1p(cv*cv);
    double z=normal(state);
    const double rho=selected->value("baseline_logvariance_correlation",0.);
    if (rho!=0.) {
      if (baseline_sigma==0.) throw std::runtime_error("Correlated noise state requires nonzero baseline scatter");
      z=rho*zbaseline+std::sqrt(1-rho*rho)*z;
    }
    noise_sigma *= std::sqrt(std::exp(std::sqrt(sigma2)*z-.5*sigma2));
  }
  for (std::size_t i = 0; i < n; ++i) {
    double noise = 0.;
    for (std::size_t k = 0; k < coefficients.size(); ++k) noise += coefficients[k]*white[i+coefficients.size()-1-k];
    const double analog = result.signal_mV[i]+baseline+noise_sigma*noise;
    const bool clipped = analog < readout.adc_min_mV || analog >= readout.adc_max_mV;
    result.analog_mV[i] = analog;
    result.adc_counts[i] = static_cast<unsigned>(std::clamp(std::floor((analog-readout.adc_min_mV)/lsb),0.,double(levels-1)));
    result.adc_clipped[i] = clipped; result.n_clipped_samples += clipped;
  }
}
