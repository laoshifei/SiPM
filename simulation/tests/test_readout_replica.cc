#include "ReadoutReplica.hh"
#include "ReadoutSampling.hh"
#include <fstream>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

void check(bool v,const char* s){if(!v)throw std::runtime_error(s);}
int main(int argc,char** argv) {
  std::ifstream file(argv[1]);nlohmann::json j;file>>j;
  auto cfg=sipm::Config::FromJSON(j.dump());auto& ro=cfg.readout;
  ro.internal_dt_ns=.5;ro.sample_period_ns=2.;ro.aperture_ns=2.;ro.lowpass_ns=110.;ro.noise_rms_mV=.1;
  sipm::Result original;original.event_id=7;original.readout_seed=75291;
  original.avalanche_time_ns={31.234,75.3,160.9};original.avalanche_charge_C={1e-13,2e-13,1e-13};original.charge_total_C=4e-13;
  nlohmann::json c;
  c["electronics_noise"]={{"model","gaussian_fir"},{"coefficients",{.6,.8}},{"event_variance_cv",.3}};
  c["signal_dependent_noise"]={{"model","gaussian_fir_signal_variance"},{"coefficients",{.8,.6}},
      {"variance_per_signal_adc",.2},{"sample_period_ns",2.},{"event_variance_cv",.2}};
  auto native=original;ReadoutSampling::Build(c,cfg,200.,0.,native);
  auto base_config=c;base_config.erase("signal_dependent_noise");
  auto cached=original;ReadoutSampling::Build(base_config,cfg,200.,0.,cached);
  for(const std::string scope:{"all_readout","signal_only","electronics_only"}) {
    for(unsigned id:{0,1,2}) {
      const ReadoutReplica control({{"noise_replica",id},{"noise_scope",scope}});
      auto a=original,b=original,again=original;
      control.Build(c,cfg,200.,0.,a);control.Build(c,cfg,200.,0.,b,&cached);control.Build(c,cfg,200.,0.,again);
      check(a.adc_counts==b.adc_counts && a.analog_mV==b.analog_mV,"Cached replica differs from uncached");
      check(a.adc_counts==again.adc_counts && a.analog_mV==again.analog_mV,"Replica is not reproducible");
      check(a.signal_mV==native.signal_mV,"Replica changed noiseless signal");
      check(a.avalanche_time_ns==original.avalanche_time_ns && a.avalanche_charge_C==original.avalanche_charge_C,"Replica changed avalanche history");
      if(id==0)check(a.adc_counts==native.adc_counts && a.analog_mV==native.analog_mV,"Zero replica changed original response");
      else check(a.analog_mV!=native.analog_mV,"Positive replica reused original noise");
    }
  }
  // Exact stream isolation, including the signal variance multiplier.
  auto all=original,sig=original,elec=original,b1=original;
  ReadoutReplica({{"noise_replica",1}}).Build(c,cfg,200.,0.,all);
  ReadoutReplica({{"noise_replica",1},{"noise_scope","signal_only"}}).Build(c,cfg,200.,0.,sig);
  ReadoutReplica({{"noise_replica",1},{"noise_scope","electronics_only"}}).Build(c,cfg,200.,0.,elec);
  ReadoutReplica({{"noise_replica",1}}).Build(base_config,cfg,200.,0.,b1);
  for(std::size_t i=0;i<all.analog_mV.size();++i) {
    check(std::abs((all.analog_mV[i]-b1.analog_mV[i])-(sig.analog_mV[i]-cached.analog_mV[i]))<1e-12,"Signal-only stream isolation failed");
    check(std::abs((elec.analog_mV[i]-b1.analog_mV[i])-(native.analog_mV[i]-cached.analog_mV[i]))<1e-12,"Electronics-only stream isolation failed");
  }
  // The cached fallback also regenerates vendor electronics when no FIR model exists.
  c.erase("electronics_noise");base_config=c;base_config.erase("signal_dependent_noise");
  cached=original;ReadoutSampling::Build(base_config,cfg,200.,1.,cached);
  auto a=original,b=original;const ReadoutReplica one({{"noise_replica",1}});
  one.Build(c,cfg,200.,1.,a);one.Build(c,cfg,200.,1.,b,&cached);
  check(a.adc_counts==b.adc_counts && a.analog_mV==b.analog_mV,"Vendor-noise/phase cache fallback differs");
  std::set<std::uint64_t> seeds;
  for(unsigned id=0;id<128;++id)check(seeds.insert(ReadoutReplica({{"noise_replica",id}}).BaselineSeed(original.readout_seed)).second,"Replica seeds collide");
  for(const auto& bad:std::vector<nlohmann::json>{{{"noise_replica",-1}},{{"noise_replica",.5}},{{"noise_replica",4294967296ULL}},{{"noise_scope","invalid"}}}) {
    bool rejected=false;try {ReadoutReplica invalid(bad);}catch(const std::exception&){rejected=true;}
    check(rejected,"Invalid replica control accepted");
  }
  std::cout<<"Noise replicas passed: exact zero identity, reproducibility, unchanged signal/avalanches, isolated streams and cached fallback\n";
}
