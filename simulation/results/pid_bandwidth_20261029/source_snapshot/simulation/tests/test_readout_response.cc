#include "ReadoutResponse.hh"
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>

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
  std::cout << "5 parallel-path checks passed: zero replay, avalanche invariance, mixture/noise, finite-window charge, validation\n";
}
