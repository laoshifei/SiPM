#ifndef GAGG_SIPM_RESPONSE_HH
#define GAGG_SIPM_RESPONSE_HH

// Numerical core. All units are explicit: ns, mm, eV, C, A and mV.
// It never uses the Geant4 transport random engine or particle truth labels.
#include <cstdint>
#include <string>
#include <vector>

namespace sipm {
constexpr double electron_charge_C = 1.602176634e-19;
constexpr double hc_eV_nm = 1239.8419843320026;

struct Sensor {
  std::string name, parameter_status, parameter_source, pde_mode;
  int cells_x, cells_y;
  double temperature_C, x_min_mm, x_max_mm, y_min_mm, y_max_mm;
  double overvoltage_V, gain_electrons, recovery_ns, trigger_scale_V, trigger_threshold_V;
  bool recovery_enabled;
  double constant_pde = 0.;
  std::vector<double> pde_wavelength_nm, pde_overvoltage_V;
  std::vector<std::vector<double>> pde_probabilities;
  double dark_rate_hz, crosstalk_probability, crosstalk_delay_ns;
  double afterpulse_probability, afterpulse_delay_ns;
  double transit_delay_mean_ns, transit_delay_sigma_ns;
  std::uint64_t max_candidates;
};
struct Readout {
  double internal_dt_ns, sample_period_ns, aperture_ns, rise_ns, decay_ns;
  double transimpedance_ohm, lowpass_ns, baseline_mV, noise_rms_mV;
  int adc_bits;
  double adc_min_mV, adc_max_mV;
  std::uint64_t max_internal_bins;
};
struct Config {
  Sensor sensor;
  Readout readout;
  std::string json_text;
  static Config FromJSON(const std::string& text);
  void ValidateWindow(double window_ns) const;
};
struct Photon {
  double time_ns, x_mm, y_mm, energy_eV;
};
enum Kind { photon = 0, dark = 1, crosstalk = 2, afterpulse = 3 };
struct Result {
  unsigned event_id = 0;
  std::uint64_t avalanche_seed = 0, readout_seed = 0;
  std::uint64_t n_input_photons = 0, n_outside_sensor = 0;
  std::uint64_t n_delayed_photons_outside = 0, n_secondary_outside = 0, n_candidates = 0;
  std::uint64_t n_photon_avalanches = 0, n_dark_avalanches = 0;
  std::uint64_t n_crosstalk_avalanches = 0, n_afterpulse_avalanches = 0, n_clipped_samples = 0;
  double charge_total_C = 0., charge_gate_C = 0.;
  double current_integral_C = 0., current_tail_C = 0., charge_balance_error_C = 0.;
  std::vector<double> avalanche_time_ns, avalanche_charge_C, avalanche_recovery;
  std::vector<int> avalanche_cell, avalanche_kind, avalanche_parent, avalanche_photon_index;
  std::vector<double> current_time_ns, current_A, sample_time_ns, signal_mV, analog_mV;
  std::vector<unsigned> adc_counts;
  std::vector<int> adc_clipped;
};

std::uint64_t EventSeed(std::uint64_t base, std::uint64_t event_id);
int MapCell(const Sensor&, double x_mm, double y_mm);
double Recovery(const Sensor&, double elapsed_ns);
double TriggerFraction(const Sensor&, double recovery);
double PhotonPDE(const Sensor&, double wavelength_nm, double recovery);
void BuildReadout(const Config&, double window_ns, Result&);
Result Simulate(const Config&, const std::vector<Photon>& sorted_photons,
                double window_ns, double gate_start_ns, double gate_end_ns,
                std::uint64_t avalanche_seed, std::uint64_t readout_seed);
}
#endif
