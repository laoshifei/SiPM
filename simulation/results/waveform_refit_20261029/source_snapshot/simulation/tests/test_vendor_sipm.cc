#include "SiPMResponse.hh"
#include <nlohmann/json.hpp>
#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace {
void Check(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
void Near(double a,double b,double tolerance,const char* message) { Check(std::abs(a-b)<=tolerance,message); }
template<class F> void Reject(F f) { bool rejected=false; try { f(); } catch(const std::exception&) { rejected=true; } Check(rejected,"invalid input was accepted"); }
}
int main(int argc,char** argv) {
  try {
    Check(argc==2,"config argument required");
    std::ifstream file(argv[1]); std::ostringstream text; text<<file.rdbuf();
    auto cfg=sipm::Config::FromJSON(text.str());
    const auto table=cfg;
    auto& s=cfg.sensor;
    s.pde_mode="constant"; s.constant_pde=1.; s.recovery_enabled=false;
    s.dark_rate_hz=0.; s.crosstalk_probability=0.; s.afterpulse_probability=0.;
    s.transit_delay_mean_ns=0.; s.transit_delay_sigma_ns=0.;
    cfg.readout.noise_rms_mV=0.;
    int passed=0;
    auto test=[&](const char* name,auto f) { f(); ++passed; std::cout<<"PASS "<<name<<'\n'; };
    test("cell boundaries",[&] {
      Check(s.name=="Hamamatsu_S13360-6050PE" && s.parameter_status=="datasheet_based","named device fixture");
      Check(sipm::MapCell(s,-3.,-3.)==0,"lower edge");
      Check(sipm::MapCell(s,3.,3.)==14399,"upper edge");
      Check(sipm::MapCell(s,3.001,0.)==-1,"outside edge");
    });
    test("bilinear PDE and no extrapolation",[&] {
      Near(sipm::PhotonPDE(table.sensor,500.,1.),.382,1e-15,"nominal PDE");
      Near(sipm::PhotonPDE(table.sensor,512.5,2.5/3.),(.300825+.2811375+.382+.357)/4.,1e-15,"interpolated PDE");
      Near(sipm::PhotonPDE(table.sensor,500.,0.),0.,0.,"zero-voltage PDE");
      Reject([&]{sipm::PhotonPDE(table.sensor,700.,1.);});
    });
    test("recovery and trigger",[&] {
      Near(sipm::Recovery(table.sensor,29.*std::log(2.)),.5,1e-15,"half recovery");
      Near(sipm::TriggerFraction(table.sensor,0.),0.,0.,"zero recovery trigger");
      Near(sipm::TriggerFraction(table.sensor,1.),1.,0.,"full recovery trigger");
    });
    test("single electron charge and fractional-bin integral",[&] {
      auto out=sipm::Simulate(cfg,{{.125,0.,0.,2.5},{19.999,0.,0.,2.5}},20.,0.,20.,31,41);
      Near(out.charge_total_C,2.*sipm::electron_charge_C*s.gain_electrons,1e-26,"avalanche charge");
      Near(out.charge_balance_error_C,0.,1e-26,"area plus tail must equal avalanche charge");
      Check(out.current_tail_C>0. && out.adc_counts.size()==10,"finite window tail and samples");
      const double tail=sipm::electron_charge_C*s.gain_electrons*(20.*std::exp(-19.875/20.)-2.*std::exp(-19.875/2.))/18.;
      const double first_q=sipm::electron_charge_C*s.gain_electrons*(20.*(-std::expm1(-.125/20.))-2.*(-std::expm1(-.125/2.)))/18.;
      Near(out.current_A[0]*.25e-9,first_q,1e-28,"analytic first bin");
      Check(out.current_tail_C>tail,"late avalanche tail");
    });
    test("empty input still produces baseline ADC",[&] {
      const auto out=sipm::Simulate(cfg,{},100.,0.,100.,32,42);
      Check(out.charge_total_C==0. && out.adc_counts.size()==50,"empty event output");
      for(auto code:out.adc_counts) Check(code==40,"baseline quantization");
    });
    test("gate uses avalanche charge and half-open endpoints",[&] {
      const auto out=sipm::Simulate(cfg,{{2.,0.,0.,2.5},{4.,0.,0.,2.5}},20.,2.,4.,33,43);
      Near(out.charge_gate_C,sipm::electron_charge_C*s.gain_electrons,1e-27,"gate selection");
      Reject([&]{sipm::Simulate(cfg,{{20.,0.,0.,2.5}},20.,0.,20.,33,43);});
    });
    test("PDE binomial statistics",[&] {
      auto c=cfg; c.sensor.constant_pde=.35;
      std::vector<sipm::Photon> hits(20000,{5.,0.,0.,2.5});
      auto out=sipm::Simulate(c,hits,100.,0.,100.,34,44);
      Near(out.n_photon_avalanches,7000.,6.*std::sqrt(20000.*.35*.65),"binomial mean");
    });
    test("same-time cell saturation",[&] {
      auto c=cfg; c.sensor.recovery_enabled=true;
      std::vector<sipm::Photon> hits(1000,{5.,0.,0.,2.5});
      auto out=sipm::Simulate(c,hits,100.,0.,100.,35,45);
      Check(out.n_photon_avalanches==1,"one discharge per cell at the same time");
    });
    test("dark count Poisson statistics",[&] {
      auto c=cfg; c.sensor.dark_rate_hz=1e8;
      const auto out=sipm::Simulate(c,{},10000.,0.,10000.,36,46);
      Near(out.n_dark_avalanches,1000.,6.*std::sqrt(1000.),"Poisson dark counts");
      Check(out.n_photon_avalanches==0,"dark-only input");
    });
    test("recursive secondary ancestry and charge",[&] {
      auto c=cfg; c.sensor.crosstalk_probability=.3; c.sensor.afterpulse_probability=.4; c.sensor.recovery_enabled=true;
      std::vector<sipm::Photon> hits;
      for(int i=0;i<100;++i) hits.push_back({5.,-2.975+.05*i,-2.975,2.5});
      const auto out=sipm::Simulate(c,hits,1000.,0.,1000.,37,47);
      Check(out.n_crosstalk_avalanches>0 && out.n_afterpulse_avalanches>0,"secondary generation");
      for(std::size_t i=0;i<out.avalanche_time_ns.size();++i) {
        const int parent=out.avalanche_parent[i];
        Check(parent<static_cast<int>(i),"parent precedes child");
        if(parent>=0) Check(out.avalanche_time_ns[i]>=out.avalanche_time_ns[parent],"causality");
        Near(out.avalanche_charge_C[i],sipm::electron_charge_C*s.gain_electrons*out.avalanche_recovery[i],1e-28,"recovered charge");
      }
    });
    test("independent reproducible electronics RNG",[&] {
      auto c=table;
      std::vector<sipm::Photon> hits(1000,{5.,0.,0.,2.5});
      const auto a=sipm::Simulate(c,hits,1000.,0.,1000.,38,48);
      const auto b=sipm::Simulate(c,hits,1000.,0.,1000.,38,49);
      const auto d=sipm::Simulate(c,hits,1000.,0.,1000.,38,48);
      Check(a.avalanche_time_ns==b.avalanche_time_ns && a.avalanche_charge_C==b.avalanche_charge_C,"readout seed changed sensor");
      Check(a.adc_counts!=b.adc_counts && a.adc_counts==d.adc_counts,"electronics reproducibility");
    });
    test("ADC clipping explicitly marked",[&] {
      auto c=cfg; c.readout.baseline_mV=1001.;
      const auto out=sipm::Simulate(c,{},100.,0.,100.,39,49);
      Check(out.n_clipped_samples==50,"upper clipping count");
      for(auto code:out.adc_counts) Check(code==4095,"upper rail");
      c.readout.baseline_mV=-1.;
      const auto low=sipm::Simulate(c,{},100.,0.,100.,39,49);
      for(auto code:low.adc_counts) Check(code==0,"lower rail");
    });
    test("invalid config/input and candidate limit reject",[&] {
      auto j=nlohmann::json::parse(text.str()); j["readout"]["model"]="measured_voltage";
      Reject([&]{sipm::Config::FromJSON(j.dump());});
      j=nlohmann::json::parse(text.str()); j["sensor"]["pde_probabilities"][0][0]=.01;
      Reject([&]{sipm::Config::FromJSON(j.dump());});
      Reject([&]{cfg.ValidateWindow(10.1);});
      Reject([&]{sipm::Simulate(cfg,{{2.,0.,0.,2.5},{1.,0.,0.,2.5}},20.,0.,20.,1,2);});
      auto c=cfg; c.sensor.max_candidates=1;
      Reject([&]{sipm::Simulate(c,{{1.,0.,0.,2.5},{2.,0.,0.,2.5}},20.,0.,20.,1,2);});
    });
    test("readout time-step convergence",[&] {
      const std::vector<sipm::Photon> hits={{10.123,0.,0.,2.5},{15.876,1.,1.,2.5}};
      const auto coarse=sipm::Simulate(cfg,hits,200.,0.,200.,1,2);
      auto c=cfg; c.readout.internal_dt_ns=.125;
      const auto fine=sipm::Simulate(c,hits,200.,0.,200.,1,2);
      double maximum=0.,error=0.;
      for(std::size_t i=0;i<fine.signal_mV.size();++i) {
        maximum=std::max(maximum,std::abs(fine.signal_mV[i]));
        error=std::max(error,std::abs(fine.signal_mV[i]-coarse.signal_mV[i]));
      }
      Check(error/maximum<.002,"0.25 ns and 0.125 ns readout differ by >=0.2% peak");
    });
    std::cout<<passed<<" tests passed\n";
  } catch(const std::exception& e) { std::cerr<<"FAIL: "<<e.what()<<'\n'; return 1; }
}
