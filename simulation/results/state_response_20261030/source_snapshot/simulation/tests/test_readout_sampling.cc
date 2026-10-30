#include "ReadoutSampling.hh"
#include "ReadoutResponse.hh"
#include "ReadoutNoise.hh"
#include "SignalDependentNoise.hh"
#include <fstream>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <limits>

void check(bool v,const char* s){if(!v)throw std::runtime_error(s);}
int main(int argc,char** argv) {
  std::ifstream file(argv[1]);nlohmann::json j;file>>j;
  auto cfg=sipm::Config::FromJSON(j.dump());auto& r=cfg.readout;
  r.internal_dt_ns=.5;r.sample_period_ns=2.;r.aperture_ns=2.;r.lowpass_ns=0.;r.noise_rms_mV=0.;
  sipm::Result base;base.avalanche_time_ns={31.234,75.3,160.9};base.avalanche_charge_C={1e-13,2e-13,1e-13};
  base.charge_total_C=4e-13;base.readout_seed=75291;
  nlohmann::json c=nlohmann::json::object();
  auto direct=base;sipm::BuildReadout(cfg,200.,direct);
  auto zero=base;ReadoutSampling::Build(c,cfg,200.,0.,zero);
  check(zero.adc_counts==direct.adc_counts && zero.signal_mV==direct.signal_mV,"Zero phase changed native ADC");
  auto cdf=[&](double t){return t<=0.?0.:1.-(r.decay_ns*std::exp(-t/r.decay_ns)-r.rise_ns*std::exp(-t/r.rise_ns))/(r.decay_ns-r.rise_ns);};
  for(double phase:{.5,1.,1.5}) {
    auto result=base;ReadoutSampling::Build(c,cfg,200.,phase,result);
    check(result.avalanche_time_ns==base.avalanche_time_ns && result.avalanche_charge_C==base.avalanche_charge_C,"Phase changed avalanches");
    for(std::size_t i=0;i<100;++i) {
      double q=0.;const double start=2*i+phase;
      for(std::size_t a=0;a<3;++a)q+=base.avalanche_charge_C[a]*(cdf(start+2-base.avalanche_time_ns[a])-cdf(start-base.avalanche_time_ns[a]));
      const double expected=q/(2e-9)*r.transimpedance_ohm*1000.;
      check(std::abs(expected-result.signal_mV[i])<1e-10,"Offset aperture differs from analytic integral");
    }
  }
  r.lowpass_ns=110.;r.noise_rms_mV=.1;
  c["readout_response"]={{"model","parallel_paths"},{"prompt_charge_fraction",.1},{"prompt_decay_ns",5.},{"prompt_lowpass_ns",0.},{"signal_transmission",.5}};
  c["electronics_noise"]={{"model","gaussian_fir"},{"coefficients",{1.}}};
  c["signal_dependent_noise"]={{"model","gaussian_fir_signal_variance"},{"coefficients",{1.}},
      {"variance_per_signal_adc",.2},{"sample_period_ns",2.}};
  auto expected=base;auto quiet=cfg;quiet.readout.noise_rms_mV=0.;
  sipm::BuildReadout(quiet,200.,expected);ReadoutResponse::Apply(c,quiet,200.,expected);
  ReadoutNoise::Apply(c,cfg.readout,expected);SignalDependentNoise::Apply(c,cfg.readout,expected);
  auto identity=base;ReadoutSampling::Build(c,cfg,200.,0.,identity);
  check(identity.adc_counts==expected.adc_counts && identity.analog_mV==expected.analog_mV,"Noisy identity mismatch");
  c.erase("signal_dependent_noise");auto a=base,b=base;
  ReadoutSampling::Build(c,cfg,200.,0.,a);ReadoutSampling::Build(c,cfg,200.,1.,b);
  for(std::size_t i=0;i<100;++i)
    check(std::abs((a.analog_mV[i]-a.signal_mV[i])-(b.analog_mV[i]-b.signal_mV[i]))<1e-12,"Phase changed paired electronics noise");
  for(double bad:{-.5,.25,2.,std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected=false;try{auto x=base;ReadoutSampling::Build(c,cfg,200.,bad,x);}catch(const std::exception&){rejected=true;}
    check(rejected,"Invalid phase accepted");
  }
  std::cout<<"Sampling tests passed: exact native identity, analytic aperture, preserved avalanches, paired noise, bounds\n";
}
