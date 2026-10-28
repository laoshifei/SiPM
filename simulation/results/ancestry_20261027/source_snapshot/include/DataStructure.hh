#pragma once

#include "TrackInformation.hh"
#include <vector>

// Physical sensor coordinates stay in mm. Microcell/channel identifiers are
// converted to the inherited response engine's internal map only in Digitize().
struct OpticalPhotonHitData {
  double time_ns;
  double x_mm;
  double y_mm;
  double energy_eV;
  int summed_channel_index;
  int microcell_index;
};

struct WaveformEventData {
  int event_id = 0;
  int n_scintillation = 0;
  int n_optical_created = 0;
  int n_late = 0;
  double edep_MeV = 0.;
  double edep_alpha_MeV = 0.;
  double edep_electron_MeV = 0.;
  double edep_ion_MeV = 0.;
  double edep_mylar_MeV = 0.;
  double edep_air_MeV = 0.;
  double charge_C = 0.;
  double charge_balance_C = 0.;
  int approximate_trigger = 0;
  std::vector<double> edep_by_origin_MeV = std::vector<double>(DecayOriginCount, 0.);
  std::vector<int> scintillation_by_origin = std::vector<int>(DecayOriginCount, 0);
  std::vector<double> charge_by_origin_C = std::vector<double>(DecayOriginCount, 0.);

  std::vector<int> decay_pdg;
  std::vector<int> decay_parent_id;
  std::vector<double> decay_energy_MeV;
  std::vector<double> decay_time_ns;
  std::vector<double> photon_t_ns;
  std::vector<double> photon_x_mm;
  std::vector<double> photon_y_mm;
  std::vector<double> photon_e_eV;
  std::vector<int> photon_origin;
  std::vector<int> photon_emitter_pdg;
  std::vector<double> avalanche_t_ns;
  std::vector<double> avalanche_q_C;
  std::vector<int> avalanche_origin;
  std::vector<int> avalanche_emitter_pdg;
  std::vector<double> signal_mV;
  std::vector<double> analog_mV;
  std::vector<unsigned> adc;
  std::vector<OpticalPhotonHitData> photon_hits;

  void Reset(int event);
};
