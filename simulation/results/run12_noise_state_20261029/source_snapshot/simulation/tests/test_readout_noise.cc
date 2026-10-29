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
    std::cout<<"PASS parameter rejection / requantization / clipping\n";
    cfg.readout.noise_rms_mV=.1; cfg.readout.baseline_mV=10.;
    model["electronics_noise"]["event_variance_cv"]=.4;
    model["electronics_noise"]["baseline_jitter_mV"]=.2;
    std::vector<double> variances, means;
    for (unsigned event=0; event<6000; ++event) {
      sipm::Result result; result.signal_mV.assign(128,0.);
      result.readout_seed=sipm::EventSeed(291029, event);
      ReadoutNoise::Apply(model,cfg.readout,result);
      variances.push_back(Variance(result.analog_mV,0,128)*128./127.);
      means.push_back(Mean(result.analog_mV,0,128));
      if (event==0) {
        auto repeated=result; ReadoutNoise::Apply(model,cfg.readout,repeated);
        Check(repeated.adc_counts==result.adc_counts,"State-scattered noise must replay exactly");
        Check(result.signal_mV==std::vector<double>(128,0.),"State noise altered noiseless signal");
      }
    }
    const double vmean=Mean(variances,0,variances.size());
    const double vvar=Variance(variances,0,variances.size());
    Check(std::abs(vmean/.01-1)<.025,"Mean-one variance state failed");
    const double expected_relative_variance=(1.+.4*.4)*(1.+2./127.)-1.;
    Check(std::abs(vvar/(vmean*vmean)/expected_relative_variance-1)<.08,"Variance-state second moment failed");
    Check(std::abs(Mean(means,0,means.size())-10.)<.01,"Baseline state mean failed");
    Check(std::abs(Variance(means,0,means.size())/(.04+.01/128.)-1)<.05,"Baseline state variance failed");
    model["electronics_noise"]["event_variance_cv"]=-1.; rejected=false;
    try {ReadoutNoise::Apply(model,cfg.readout,signal);} catch (const std::runtime_error&) {rejected=true;}
    Check(rejected,"Negative variance-state CV must be rejected");
    std::cout<<"PASS event variance / baseline state moments / reproducibility\n";
    model["electronics_noise"]["event_variance_cv"]=0.;
    model["electronics_noise"]["states"]={
      {{"weight",.3},{"baseline_mV",10.},{"noise_rms_mV",.1},{"baseline_jitter_mV",.2},
       {"event_variance_cv",.4},{"baseline_logvariance_correlation",.6}},
      {{"weight",.7},{"baseline_mV",20.},{"noise_rms_mV",.2},{"baseline_jitter_mV",.1},
       {"event_variance_cv",.1},{"baseline_logvariance_correlation",0.}}};
    unsigned first_count=0;double sx=0.,sy=0.,sxx=0.,syy=0.,sxy=0.;
    for (unsigned event=0;event<6000;++event) {
      sipm::Result result;result.signal_mV.assign(128,0.);result.readout_seed=sipm::EventSeed(291029,event);
      ReadoutNoise::Apply(model,cfg.readout,result);
      const double x=Mean(result.analog_mV,0,128),y=std::log(Variance(result.analog_mV,0,128));
      if (x<15.) {++first_count;sx+=x;sy+=y;sxx+=x*x;syy+=y*y;sxy+=x*y;}
    }
    Check(std::abs(double(first_count)/6000-.3)<.025,"State-mixture weights failed");
    const double corr=(sxy-sx*sy/first_count)/std::sqrt((sxx-sx*sx/first_count)*(syy-sy*sy/first_count));
    Check(corr>.45 && corr<.70,"Baseline/logvariance correlation failed");
    model["electronics_noise"]["states"][0]["weight"]=.8;rejected=false;
    try {ReadoutNoise::Apply(model,cfg.readout,signal);} catch (const std::runtime_error&) {rejected=true;}
    Check(rejected,"Non-normalized state weights must be rejected");
    std::cout<<"PASS baseline-state mixture / covariance / weight validation\n5 tests passed\n";
    return 0;
  } catch (const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
