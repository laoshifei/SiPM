#include "ReadoutResponse.hh"
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
#include <limits>

void check(bool ok, const char* text) { if (!ok) throw std::runtime_error(text); }
int main(int argc, char** argv) {
  std::ifstream input(argv[1]); nlohmann::json j; input >> j;
  auto cfg = sipm::Config::FromJSON(j.dump());
  cfg.readout.lowpass_ns = 94.;
  sipm::Result slow;
  slow.avalanche_time_ns = {1.234, 79.6, 990.1};
  slow.avalanche_charge_C = {2e-13, 1e-13, 2e-13};
  slow.charge_total_C = 5e-13; slow.readout_seed = 123;
  sipm::BuildReadout(cfg, 1000., slow);
  nlohmann::json c = {{"readout_response", {{"model", "parallel_paths"}, {"prompt_charge_fraction", 0.}, {"prompt_decay_ns", 5.}, {"prompt_lowpass_ns", 0.}}}};
  auto zero = slow; ReadoutResponse::Apply(c, cfg, 1000., zero);
  check(zero.adc_counts == slow.adc_counts && zero.signal_mV == slow.signal_mV, "Zero fraction changed legacy output");
  c["readout_response"]["signal_transmission"] = 1.;
  auto unit = slow; ReadoutResponse::Apply(c, cfg, 1000., unit);
  check(unit.adc_counts == slow.adc_counts && unit.analog_mV == slow.analog_mV, "Unit transfer changed legacy output");
  c["readout_response"]["signal_transmission"] = .5;
  auto attenuated = slow; ReadoutResponse::Apply(c, cfg, 1000., attenuated);
  check(attenuated.avalanche_charge_C == slow.avalanche_charge_C && attenuated.current_A == slow.current_A,
        "Voltage transfer changed physical SiPM charge/current");
  check(attenuated.current_integral_C == slow.current_integral_C && attenuated.current_tail_C == slow.current_tail_C,
        "Voltage transfer changed physical charge accounting");
  for (std::size_t i=0; i<slow.signal_mV.size(); ++i) {
    check(attenuated.signal_mV[i] == .5*slow.signal_mV[i], "Incorrect voltage attenuation");
    check(std::abs((attenuated.analog_mV[i]-attenuated.signal_mV[i])-(slow.analog_mV[i]-slow.signal_mV[i]))<1e-12,
          "Voltage transfer attenuated downstream noise");
  }
  for (double bad : {0., -1., 1.1, std::numeric_limits<double>::infinity()}) {
    c["readout_response"]["signal_transmission"] = bad;
    bool rejected=false; try { ReadoutResponse::Validate(c,cfg); } catch (const std::exception&) { rejected=true; }
    check(rejected,"Invalid transmission accepted");
  }
  c["readout_response"]["signal_transmission"] = 1.;
  auto fastcfg = cfg; fastcfg.readout.decay_ns = 5.; fastcfg.readout.lowpass_ns = 0.;
  auto fast = slow; sipm::BuildReadout(fastcfg, 1000., fast);
  c["readout_response"]["prompt_charge_fraction"] = .2;
  auto mixed = slow; ReadoutResponse::Apply(c, cfg, 1000., mixed);
  check(mixed.avalanche_time_ns == slow.avalanche_time_ns && mixed.avalanche_charge_C == slow.avalanche_charge_C, "Response changed avalanche history");
  for (std::size_t i=0; i<mixed.signal_mV.size(); ++i) {
    check(std::abs(mixed.signal_mV[i]-(.8*slow.signal_mV[i]+.2*fast.signal_mV[i])) < 1e-12, "Wrong path mixture");
    check(std::abs((mixed.analog_mV[i]-mixed.signal_mV[i])-(slow.analog_mV[i]-slow.signal_mV[i])) < 1e-12, "Noise realization changed");
  }
  check(std::abs(mixed.charge_balance_error_C)/mixed.charge_total_C < 1e-10, "Finite-window charge not conserved");
  c["readout_response"]["prompt_charge_fraction"] = 1.1;
  bool rejected = false; try { ReadoutResponse::Validate(c, cfg); } catch (const std::exception&) { rejected=true; }
  check(rejected, "Invalid charge fraction accepted");
  std::cout << "Parallel paths and voltage transfer checks passed: legacy/unit replay, physical charge/current, downstream noise, mixture, bounds\n";
}
