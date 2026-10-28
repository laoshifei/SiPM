#include "DataStructure.hh"
#include <algorithm>

void WaveformEventData::Reset(int event) {
  event_id = event;
  n_scintillation = n_optical_created = n_late = 0;
  edep_MeV = edep_alpha_MeV = edep_electron_MeV = edep_ion_MeV = 0.;
  edep_mylar_MeV = edep_air_MeV = 0.;
  charge_C = charge_balance_C = 0.;
  approximate_trigger = 0;
  std::fill(edep_by_origin_MeV.begin(), edep_by_origin_MeV.end(), 0.);
  std::fill(edep_origin_species_MeV.begin(), edep_origin_species_MeV.end(), 0.);
  std::fill(scintillation_by_origin.begin(), scintillation_by_origin.end(), 0);
  std::fill(charge_by_origin_C.begin(), charge_by_origin_C.end(), 0.);
  decay_pdg.clear();
  decay_parent_id.clear();
  decay_energy_MeV.clear();
  decay_time_ns.clear();
  photon_t_ns.clear();
  photon_x_mm.clear();
  photon_y_mm.clear();
  photon_e_eV.clear();
  photon_origin.clear();
  photon_emitter_pdg.clear();
  photon_boundary_status.clear();
  avalanche_t_ns.clear();
  avalanche_q_C.clear();
  avalanche_origin.clear();
  avalanche_emitter_pdg.clear();
  signal_mV.clear();
  analog_mV.clear();
  adc.clear();
  photon_hits.clear();
}
