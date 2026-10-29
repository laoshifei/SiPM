#include "RootIO.hh"
#include "G4OpticalParameters.hh"
#include "RunConfig.hh"
#include "DataStructure.hh"
#include "G4Version.hh"
#include <TFile.h>
#include <TTree.h>
#include <TObjString.h>
#include <filesystem>
#include <stdexcept>
#include <utility>

RootIO::RootIO(const RunConfig& config, std::shared_ptr<WaveformEventData> data)
    : config_(config), data_(std::move(data)) {}

RootIO::~RootIO() {
  // Exceptional/aborted runs retain completed=false in their initial metadata.
  ReleaseFile();
}

void RootIO::Open(int requested_events) {
  if (file_) throw std::runtime_error("ROOT output is already open");
  const std::filesystem::path path(config_.Output());
  if (std::filesystem::exists(path)) throw std::runtime_error("Refusing to overwrite output: " + path.string());
  if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
  file_ = std::make_unique<TFile>(path.string().c_str(), "CREATE");
  if (file_->IsZombie()) throw std::runtime_error("Cannot create ROOT output");
  tree_ = std::make_unique<TTree>("Events", "One parent source decay; all saved including zero deposits");
  tree_->SetDirectory(file_.get());
  tree_->SetAutoFlush(-32 * 1024 * 1024);

#define BRANCH(field) tree_->Branch(#field, &data_->field)
  BRANCH(event_id);
  BRANCH(n_scintillation);
  BRANCH(n_optical_created);
  BRANCH(n_late);
  BRANCH(edep_MeV);
  BRANCH(edep_alpha_MeV);
  BRANCH(edep_electron_MeV);
  BRANCH(edep_ion_MeV);
  BRANCH(edep_mylar_MeV);
  BRANCH(edep_air_MeV);
  BRANCH(edep_by_origin_MeV);
  BRANCH(edep_origin_species_MeV);
  BRANCH(scintillation_by_origin);
  BRANCH(charge_by_origin_C);
  BRANCH(charge_C);
  BRANCH(charge_balance_C);
  BRANCH(approximate_trigger);
  BRANCH(decay_pdg);
  BRANCH(decay_parent_id);
  BRANCH(decay_energy_MeV);
  BRANCH(decay_time_ns);
  BRANCH(photon_t_ns);
  BRANCH(photon_x_mm);
  BRANCH(photon_y_mm);
  BRANCH(photon_e_eV);
  BRANCH(photon_origin);
  BRANCH(photon_emitter_pdg);
  BRANCH(photon_boundary_status);
  BRANCH(avalanche_t_ns);
  BRANCH(avalanche_q_C);
  BRANCH(avalanche_origin);
  BRANCH(avalanche_emitter_pdg);
  BRANCH(signal_mV);
  BRANCH(analog_mV);
  BRANCH(adc);
#undef BRANCH

  metadata_ = {
      {"schema_version", 9}, {"source", config_.Source()}, {"events_requested", requested_events},
      {"seed", config_.Seed()}, {"geant4", G4Version}, {"config", config_.Data()},
      {"completed", false}, {"config_file", config_.ConfigFile()},
      {"digitization_enabled", config_.Acquisition().value("digitize", true)},
      {"electronics_noise", config_.Data().value("electronics_noise", nlohmann::json{{"model", "independent_gaussian"}})},
      {"readout_response", config_.Data().value("readout_response", nlohmann::json{{"model", "legacy_single_path"}})},
      {"signal_dependent_noise", config_.Data().value("signal_dependent_noise", nlohmann::json{{"model", "disabled"}})},
      {"scintillation_time_model", {
          {"finite_rise_enabled", G4OpticalParameters::Instance()->GetScintFiniteRiseTime()},
          {"rise_time_ns", config_.Optical().value("rise_time_ns", std::vector<double>{0., 0.})},
          {"definition", "Geant4 finite rise per component: normalized (1-exp(-t/rise))*exp(-t/decay); zero rise gives exponential. Rise constants shared by all particle types; decay constants remain particle dependent."}}},
      {"architecture", "DetectorConstruction/PhysicsList/ActionInitialization; G4DigiManager SiPMDigitizer"},
      {"optical_hit_collection", "SiPM/OpticalArrivals; entrance steps forwarded to SiPMSD by SteppingAction; photon_boundary_status uses G4OpBoundaryProcessStatus codes"},
      {"sensor_entry_policy", "Only Transmission(1), FresnelRefraction(2), SameMaterial(12) are accepted; reflected tracks continue. Supersedes legacy post-volume scoring in schema <=3."},
      {"edep_origin_species_definition", "CsI deposition in flattened [6 ancestry categories][alpha, e-/e+, other]. Direct species, no scintillation yield or optical response applied."},
      {"decay_products", "All in-window particles born through RadioactiveDecay, including prompt daughter de-excitations; not all secondary tracks"},
      {"parent_clock", "Parent lifetime set to zero; conditional isolated decay at pretrigger. Nuclear branches unchanged; no absolute activity or exposure."},
      {"particle_truth", "Simulation only; no experimental truth inferred"},
      {"origin_categories", {"unknown", "alpha_descendants", "gamma_descendants", "decay_electron_descendants", "recoil_descendants", "other"}},
      {"origin_definition", "Ancestry resets at each RadioactiveDecay product, including prompt daughter de-excitation. All transport descendants inherit it. Decay electrons include beta/conversion/atomic-relaxation electrons; optical emitter PDG records the immediate scintillation/Cerenkov parent. This diagnostic does not alter the light-yield model."},
      {"avalanche_origin_definition", "Photon avalanches inherit photon ancestry; AP/XT descendants inherit their triggering avalanche ancestry; primary dark avalanches use unknown=0. charge_by_origin_C sums realized in-window avalanche charge, not the finite-gate ADC integral."},
      {"trigger", "Approximate amplitude flag only; all events saved, PHA hardware trigger NOT emulated"},
      {"limitations", {
          "Scintillation yield ratios are constant hypotheses; no Birks double correction",
          "Sensor entrance records optical photons; direct ionizing SiPM charge is not modeled",
          "Optional planar source-exit foil is a declared scenario, not a confirmed capsule or backing",
          "Uniform effective microcell mapping; XT/AP disabled by default"}}
  };
  TObjString(metadata_.dump().c_str()).Write("RunInfo");
}

void RootIO::FillEvent() {
  if (!tree_) throw std::runtime_error("RunAction has not opened the ROOT event tree");
  if (tree_->Fill() < 0) throw std::runtime_error("Cannot write ROOT event");
}

void RootIO::Close(int completed_events) {
  if (!file_ || !tree_) throw std::runtime_error("No open ROOT run to close");
  if (tree_->GetEntries() != completed_events) throw std::runtime_error("Saved/processed event count mismatch");
  metadata_["events_completed"] = completed_events;
  metadata_["completed"] = completed_events == metadata_.at("events_requested").get<int>();
  file_->cd();
  if (tree_->Write() <= 0) throw std::runtime_error("Cannot finalize ROOT event tree");
  if (TObjString(metadata_.dump().c_str()).Write("RunInfo", TObject::kOverwrite) <= 0) {
    throw std::runtime_error("Cannot finalize ROOT run metadata");
  }
  ReleaseFile();
}

void RootIO::ReleaseFile() {
  if (tree_) tree_->SetDirectory(nullptr);
  tree_.reset();
  if (file_) file_->Close();
  file_.reset();
}
