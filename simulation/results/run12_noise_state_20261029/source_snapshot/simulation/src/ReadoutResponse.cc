#include "ReadoutResponse.hh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

bool ReadoutResponse::Enabled(const nlohmann::json& config) {
  return config.contains("readout_response");
}

void ReadoutResponse::Validate(const nlohmann::json& config, const sipm::Config& sensor) {
  if (!Enabled(config)) return;
  const auto& model = config.at("readout_response");
  if (model.at("model") != "parallel_paths") throw std::runtime_error("Unknown readout response model");
  const double f = model.at("prompt_charge_fraction");
  const double decay = model.at("prompt_decay_ns");
  const double lowpass = model.at("prompt_lowpass_ns");
  const double transmission = model.value("signal_transmission",1.);
  if (!std::isfinite(f) || f < 0. || f > 1. || !std::isfinite(decay)
      || decay <= sensor.readout.rise_ns || !std::isfinite(lowpass) || lowpass < 0.)
    throw std::runtime_error("Invalid parallel readout response parameters");
  if (!std::isfinite(transmission) || transmission <= 0. || transmission > 1.)
    throw std::runtime_error("Signal transmission must be finite and in (0,1]");
}

void ReadoutResponse::Apply(const nlohmann::json& config, const sipm::Config& sensor,
                            double duration, sipm::Result& result) {
  if (!Enabled(config)) return;
  Validate(config, sensor);
  const auto& model = config.at("readout_response");
  const double f = model.at("prompt_charge_fraction");
  const double transmission = model.value("signal_transmission",1.);
  if (f == 0. && transmission == 1.) return;  // Exact legacy replay.
  sipm::Result fast;
  if (f > 0.) {
    auto fast_config = sensor;
    fast_config.readout.decay_ns = model.at("prompt_decay_ns");
    fast_config.readout.lowpass_ns = model.at("prompt_lowpass_ns");
    fast_config.readout.noise_rms_mV = 0.;
    fast.avalanche_time_ns = result.avalanche_time_ns;
    fast.avalanche_charge_C = result.avalanche_charge_C;
    fast.charge_total_C = result.charge_total_C;
    sipm::BuildReadout(fast_config, duration, fast);
    if (fast.signal_mV.size() != result.signal_mV.size()) throw std::runtime_error("Readout path sample mismatch");
    for (std::size_t i = 0; i < result.current_A.size(); ++i)
      result.current_A[i] = (1.-f)*result.current_A[i]+f*fast.current_A[i];
    result.current_integral_C = (1.-f)*result.current_integral_C+f*fast.current_integral_C;
    result.current_tail_C = (1.-f)*result.current_tail_C+f*fast.current_tail_C;
    result.charge_balance_error_C = result.current_integral_C+result.current_tail_C-result.charge_total_C;
  }
  const auto& r = sensor.readout;
  const unsigned levels = 1u << r.adc_bits;
  const double lsb = (r.adc_max_mV-r.adc_min_mV)/levels;
  result.n_clipped_samples = 0;
  for (std::size_t i = 0; i < result.signal_mV.size(); ++i) {
    // Preserve the original baseline/noise realization. A configured Gaussian
    // FIR is applied once afterwards by ReadoutNoise, as in the legacy path.
    const double noise = result.analog_mV[i]-result.signal_mV[i];
    // This effective transfer factor is distinct from the SiPM avalanche gain
    // and the digitizer's nominal50ohm input. Do not attenuate downstream noise.
    const double signal = transmission*(f == 0. ? result.signal_mV[i]
        : (1.-f)*result.signal_mV[i]+f*fast.signal_mV[i]);
    const double analog = signal+noise;
    result.signal_mV[i] = signal;
    result.analog_mV[i] = analog;
    const bool clipped = analog < r.adc_min_mV || analog >= r.adc_max_mV;
    result.adc_counts[i] = static_cast<unsigned>(std::clamp(std::floor((analog-r.adc_min_mV)/lsb),0.,double(levels-1)));
    result.adc_clipped[i] = clipped;
    result.n_clipped_samples += clipped;
  }
}
