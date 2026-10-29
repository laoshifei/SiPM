#include "ReadoutNoise.hh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {
void Check(bool condition, const char* message) { if (!condition) throw std::runtime_error(message); }
double Mean(const std::vector<double>& x, std::size_t first, std::size_t n) {
  return std::accumulate(x.begin()+first, x.begin()+first+n, 0.)/n;
}
double Variance(const std::vector<double>& x, std::size_t first, std::size_t n) {
  const double mean = Mean(x,first,n); double value = 0.;
  for (std::size_t i = first; i < first+n; ++i) value += (x[i]-mean)*(x[i]-mean);
  return value/n;
}
}

int main(int argc, char** argv) {
  try {
    Check(argc==2,"config path required");
    std::ifstream input(argv[1]); std::ostringstream text; text<<input.rdbuf();
    auto cfg = sipm::Config::FromJSON(text.str()); cfg.readout.noise_rms_mV = .1;
    cfg.sensor.dark_rate_hz = 0.;
    auto zero = cfg; zero.readout.noise_rms_mV = 0.;
    auto original = sipm::Simulate(cfg,{},1000.,0.,1000.,33,44);
    auto signal = sipm::Simulate(zero,{},1000.,0.,1000.,33,44);
    nlohmann::json model = {{"electronics_noise", {{"model","gaussian_fir"},{"coefficients",{1.}}}}};
    ReadoutNoise::Apply(model,cfg.readout,signal);
    Check(signal.adc_counts==original.adc_counts && signal.analog_mV==original.analog_mV,"Unit FIR must reproduce vendor white-noise samples exactly");
    std::cout<<"PASS unit FIR / vendor noise RNG and ADC identity\n";
    signal = sipm::Simulate(zero,{},200000.,0.,200000.,33,44);
    const auto currents = signal.current_A; const auto voltages = signal.signal_mV;
    model["electronics_noise"]["coefficients"]={std::sqrt(.5),std::sqrt(.5)};
    ReadoutNoise::Apply(model,cfg.readout,signal); const double mean=Mean(signal.analog_mV,0,signal.analog_mV.size());
    const double var=Variance(signal.analog_mV,0,signal.analog_mV.size()); double c1=0.,c2=0.;
    for (std::size_t i=2;i<signal.analog_mV.size();++i) {
      c1+=(signal.analog_mV[i]-mean)*(signal.analog_mV[i-1]-mean);
      c2+=(signal.analog_mV[i]-mean)*(signal.analog_mV[i-2]-mean);
    }
    Check(std::abs(var/.01-1.)<.03,"FIR output variance");
    Check(std::abs(c1/(signal.analog_mV.size()-2)/var-.5)<.02,"FIR lag-one covariance");
    Check(std::abs(c2/(signal.analog_mV.size()-2)/var)<.02,"FIR lag-two covariance");
    Check(signal.current_A==currents && signal.signal_mV==voltages,"Noise must not alter current or noiseless signal");
    Check(std::abs(Variance(signal.analog_mV,0,4096)/.01-1.)<.12,"Stationary start, no zero-history transient");
    std::cout<<"PASS stationary FIR variance / covariance / preserved signal\n";
    model["electronics_noise"]["coefficients"]={2.}; bool rejected=false;
    try {ReadoutNoise::Apply(model,cfg.readout,signal);} catch (const std::runtime_error&) {rejected=true;}
    Check(rejected,"Non-unit FIR must be rejected");
    model["electronics_noise"]["coefficients"]={1.};cfg.readout.noise_rms_mV=0.;cfg.readout.baseline_mV=cfg.readout.adc_max_mV+1.;
    ReadoutNoise::Apply(model,cfg.readout,signal);
    Check(signal.n_clipped_samples==signal.adc_counts.size(),"Clipping must be counted after noise");
    std::cout<<"PASS parameter rejection / requantization / clipping\n3 tests passed\n";
    return 0;
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
