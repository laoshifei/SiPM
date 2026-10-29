#include "SignalDependentNoise.hh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace {
void Check(bool condition,const char* message) { if(!condition) throw std::runtime_error(message); }
double Mean(const std::vector<double>& x) { return std::accumulate(x.begin(),x.end(),0.)/x.size(); }
double Variance(const std::vector<double>& x) { double m=Mean(x),v=0.;for(double y:x)v+=(y-m)*(y-m);return v/x.size(); }
sipm::Result Constant(const sipm::Readout& r,double adc) {
  const double lsb=(r.adc_max_mV-r.adc_min_mV)/(1u<<r.adc_bits);
  sipm::Result v;v.readout_seed=31728;
  v.signal_mV.assign(100000,adc*lsb);v.analog_mV.assign(100000,r.baseline_mV+adc*lsb);
  v.adc_counts.assign(100000,unsigned(std::floor((r.baseline_mV+adc*lsb-r.adc_min_mV)/lsb)));
  v.adc_clipped.assign(100000,0);v.avalanche_time_ns={3.,7.};v.avalanche_charge_C={1e-13,2e-13};
  v.current_A={1e-5};v.charge_total_C=3e-13;return v;
}
}

int main(int argc,char** argv) {
  try {
    Check(argc==2,"Fixture required");std::ifstream input(argv[1]);std::ostringstream text;text<<input.rdbuf();
    auto cfg=sipm::Config::FromJSON(text.str());auto r=cfg.readout;
    r.baseline_mV=(r.adc_min_mV+r.adc_max_mV)/3.;
    const double lsb=(r.adc_max_mV-r.adc_min_mV)/(1u<<r.adc_bits);
    nlohmann::json j;
    j["signal_dependent_noise"]={{"model","gaussian_fir_signal_variance"},
      {"sample_period_ns",r.sample_period_ns},{"variance_per_signal_adc",.4},
      {"coefficients",{std::sqrt(.5),std::sqrt(.5)}}};
    auto base=Constant(r,100.);auto disabled=base;
    SignalDependentNoise::Apply(nlohmann::json::object(),r,disabled);
    Check(disabled.analog_mV==base.analog_mV && disabled.adc_counts==base.adc_counts,"Absent model changes legacy samples");
    auto zero=j;zero["signal_dependent_noise"]["variance_per_signal_adc"]=0.;
    SignalDependentNoise::Apply(zero,r,disabled);
    Check(disabled.analog_mV==base.analog_mV,"Zero variance is not exact no-op");
    auto empty=Constant(r,0.);const auto empty_adc=empty.adc_counts;
    SignalDependentNoise::Apply(j,r,empty);Check(empty.adc_counts==empty_adc,"Signal-free baseline changed");
    std::cout<<"PASS disabled,zero strength and zero signal identities\n";
    auto a=base,b=Constant(r,400.);SignalDependentNoise::Apply(j,r,a);SignalDependentNoise::Apply(j,r,b);
    std::vector<double> da(a.analog_mV.size()),db(da.size());
    for(std::size_t i=0;i<da.size();++i) {
      da[i]=(a.analog_mV[i]-base.analog_mV[i])/lsb;
      db[i]=(b.analog_mV[i]-(r.baseline_mV+400.*lsb))/lsb;
      Check(std::abs(db[i]-2.*da[i])<1e-9,"Variance fails linear signal scaling");
    }
    double var=Variance(da),mean=Mean(da),lag=0.;
    for(std::size_t i=1;i<da.size();++i)lag+=(da[i]-mean)*(da[i-1]-mean);
    Check(std::abs(var/40.-1.)<.03,"Conditional variance differs from specification");
    Check(std::abs(mean)<.12,"Nonzero mean residual");
    Check(std::abs(lag/(da.size()-1)/var-.5)<.02,"FIR covariance differs from analytic value");
    Check(a.signal_mV==base.signal_mV && a.current_A==base.current_A && a.avalanche_charge_C==base.avalanche_charge_C && a.charge_total_C==base.charge_total_C,"Residual changed physical signal/charge");
    std::cout<<"PASS mean,linear variance scaling,FIR covariance and preserved physical charge\n";
    auto again=base;SignalDependentNoise::Apply(j,r,again);Check(again.adc_counts==a.adc_counts,"Seed not reproducible");
    again=base;again.readout_seed++;SignalDependentNoise::Apply(j,r,again);Check(again.adc_counts!=a.adc_counts,"Seed does not change residual");
    std::cout<<"PASS deterministic independent seed stream\n";
    auto mixed=j;mixed["signal_dependent_noise"]["coefficients"]={1.};
    mixed["signal_dependent_noise"]["event_variance_cv"]=.35;
    std::vector<double> event_power;
    for(unsigned seed=0;seed<4000;++seed) {
      sipm::Result v;v.readout_seed=seed+123;
      v.signal_mV.assign(32,100.*lsb);v.analog_mV.assign(32,r.baseline_mV+100.*lsb);
      v.adc_counts.assign(32,0);
      SignalDependentNoise::Apply(mixed,r,v);double power=0.;
      for(double y:v.analog_mV) {double d=(y-r.baseline_mV-100.*lsb)/lsb;power+=d*d;}
      event_power.push_back(power/(32.*40.));
    }
    // Var(M*chi2_32/32) = CV(M)^2 + (2/32)*(1+CV(M)^2).
    Check(std::abs(Mean(event_power)-1.)<.035,"Event multiplier mean is not one");
    Check(std::abs(Variance(event_power)-(.35*.35+2./32*(1.+.35*.35)))<.03,"Lognormal variance scatter does not follow specified moments");
    std::cout<<"PASS event-scatter mean and variance including finite-waveform statistics\n";
    for(int mode=0;mode<3;++mode) {
      auto bad=j;if(mode==0)bad["signal_dependent_noise"]["variance_per_signal_adc"]=-1.;
      if(mode==1)bad["signal_dependent_noise"]["coefficients"]={2.};
      if(mode==2)bad["signal_dependent_noise"]["sample_period_ns"]=r.sample_period_ns*2;
      bool rejected=false;try{SignalDependentNoise::Apply(bad,r,again);}catch(const std::runtime_error&){rejected=true;}
      Check(rejected,"Invalid calibration accepted");
    }
    auto clipped=base;for(double& y:clipped.analog_mV)y=r.adc_max_mV+1000.;
    SignalDependentNoise::Apply(j,r,clipped);Check(clipped.n_clipped_samples==clipped.adc_counts.size(),"Clipping not recorded");
    std::cout<<"PASS invalid settings and clipping\n5 tests passed\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
