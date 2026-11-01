#include "ReadoutSampling.hh"
#include "ReadoutResponse.hh"
#include "ReadoutNoise.hh"
#include "SignalDependentNoise.hh"
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>

void ReadoutSampling::Build(const nlohmann::json& config, const sipm::Config& sensor,
                            double duration, double phase, sipm::Result& result) {
  const auto& r=sensor.readout;
  const double grid=phase/r.internal_dt_ns;
  if (!std::isfinite(phase) || phase<0. || phase>=r.sample_period_ns
      || std::abs(grid-std::round(grid))>1e-9)
    throw std::runtime_error("Sampling phase must be on the native integration grid within one ADC period");
  auto simulated=sensor;
  if (ReadoutNoise::Enabled(config)) simulated.readout.noise_rms_mV=0.;
  if (phase==0.) {
    sipm::BuildReadout(simulated,duration,result);
    ReadoutResponse::Apply(config,simulated,duration,result);
  } else {
    auto fine=simulated;
    fine.readout.sample_period_ns=r.internal_dt_ns;
    fine.readout.aperture_ns=r.internal_dt_ns;
    fine.readout.noise_rms_mV=0.;
    // Extend only the integration buffer; no new avalanche is invented beyond
    // the saved history. Analyses must avoid the censored record boundary.
    sipm::BuildReadout(fine,duration+r.sample_period_ns,result);
    ReadoutResponse::Apply(config,fine,duration+r.sample_period_ns,result);
    const auto signal=std::move(result.signal_mV);
    const auto n=static_cast<std::size_t>(std::llround(duration/r.sample_period_ns));
    const auto stride=static_cast<std::size_t>(std::llround(r.sample_period_ns/r.internal_dt_ns));
    const auto aperture=static_cast<std::size_t>(std::llround(r.aperture_ns/r.internal_dt_ns));
    const auto offset=static_cast<std::size_t>(std::llround(grid));
    result.signal_mV.resize(n);result.analog_mV.resize(n);result.sample_time_ns.resize(n);
    result.adc_counts.resize(n);result.adc_clipped.resize(n);result.n_clipped_samples=0;
    std::mt19937_64 engine(result.readout_seed);
    std::normal_distribution<double> gaussian(0.,1.);
    const unsigned levels=1u<<r.adc_bits;
    const double lsb=(r.adc_max_mV-r.adc_min_mV)/levels;
    for (std::size_t i=0;i<n;++i) {
      double sum=0.;
      for (std::size_t k=0;k<aperture;++k) sum+=signal.at(i*stride+offset+k);
      const double value=sum/aperture;
      const double analog=value+r.baseline_mV+(simulated.readout.noise_rms_mV>0.
          ? simulated.readout.noise_rms_mV*gaussian(engine):0.);
      const bool clipped=analog<r.adc_min_mV || analog>=r.adc_max_mV;
      result.sample_time_ns[i]=i*r.sample_period_ns+phase+.5*r.aperture_ns;
      result.signal_mV[i]=value;result.analog_mV[i]=analog;
      result.adc_counts[i]=static_cast<unsigned>(std::clamp(std::floor((analog-r.adc_min_mV)/lsb),0.,double(levels-1)));
      result.adc_clipped[i]=clipped;result.n_clipped_samples+=clipped;
    }
  }
  ReadoutNoise::Apply(config,sensor.readout,result);
  SignalDependentNoise::Apply(config,sensor.readout,result);
}
