#include "EventAction.hh"
#include "RunConfig.hh"
#include "DataStructure.hh"
#include "RootIO.hh"
#include "SiPMDigitizer.hh"
#include "SiPMSD.hh"
#include "G4SDManager.hh"
#include "G4HCofThisEvent.hh"
#include "G4DigiManager.hh"
#include "G4Event.hh"
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <utility>

EventAction::EventAction(const RunConfig& config, std::shared_ptr<WaveformEventData> data,
                         std::shared_ptr<RootIO> output)
    : config_(config), data_(std::move(data)), output_(std::move(output)) {}

void EventAction::BeginOfEventAction(const G4Event* event) {
  data_->Reset(event->GetEventID());
}

void EventAction::EndOfEventAction(const G4Event* event) {
  const auto hit_id = G4SDManager::GetSDMpointer()->GetCollectionID(SiPMSD::Collection);
  auto* hit_collections = event->GetHCofThisEvent();
  const auto* hits = hit_collections && hit_id >= 0
      ? dynamic_cast<const SiPMHitsCollection*>(hit_collections->GetHC(hit_id)) : nullptr;
  if (!hits) throw std::runtime_error("SiPM optical hit collection is missing");
  for (std::size_t i = 0; i < hits->entries(); ++i) {
    const auto* hit = (*hits)[i];
    const auto& photon = hit->photon;
    data_->photon_hits.push_back(photon);
    data_->photon_t_ns.push_back(photon.time_ns);
    data_->photon_x_mm.push_back(photon.x_mm);
    data_->photon_y_mm.push_back(photon.y_mm);
    data_->photon_e_eV.push_back(photon.energy_eV);
    data_->photon_origin.push_back(hit->origin);
    data_->photon_emitter_pdg.push_back(hit->emitter_pdg);
    data_->photon_boundary_status.push_back(hit->boundary_status);
  }
  auto* manager = G4DigiManager::GetDMpointer();
  auto* digitizer = dynamic_cast<SiPMDigitizer*>(manager->FindDigitizerModule(SiPMDigitizer::Name));
  if (!digitizer) throw std::runtime_error("SiPM digitizer is not registered");
  digitizer->SetInput(*data_);
  manager->Digitize(SiPMDigitizer::Name);
  const auto id = manager->GetDigiCollectionID(SiPMDigitizer::Collection);
  const auto* collection = dynamic_cast<const SiPMDigiCollection*>(manager->GetDigiCollection(id));
  if (!collection || collection->entries() != 1) {
    throw std::runtime_error("Expected exactly one SiPM waveform in the G4Event digi collection");
  }
  auto& response = (*collection)[0]->response;
  data_->charge_C = response.charge_total_C;
  data_->charge_balance_C = response.charge_balance_error_C;
  data_->avalanche_t_ns = std::move(response.avalanche_time_ns);
  data_->avalanche_q_C = std::move(response.avalanche_charge_C);
  data_->avalanche_origin = std::move((*collection)[0]->avalanche_origin);
  data_->avalanche_emitter_pdg = std::move((*collection)[0]->avalanche_emitter_pdg);
  for (std::size_t i = 0; i < data_->avalanche_q_C.size(); ++i) {
    data_->charge_by_origin_C.at(data_->avalanche_origin[i]) += data_->avalanche_q_C[i];
  }
  data_->signal_mV = std::move(response.signal_mV);
  data_->analog_mV = std::move(response.analog_mV);
  data_->adc = std::move(response.adc_counts);

  const auto& readout = config_.SensorModel().readout;
  const double scale = (readout.adc_max_mV - readout.adc_min_mV) / (1u << readout.adc_bits);
  const double threshold = config_.Acquisition().at("amplitude_threshold_adc");
  data_->approximate_trigger = *std::max_element(data_->signal_mV.begin(), data_->signal_mV.end()) / scale >= threshold;
  output_->FillEvent();
  if (data_->event_id % 100 == 0) {
    std::cout << "event " << data_->event_id << ", Edep=" << data_->edep_MeV
              << " MeV, detected photons=" << data_->photon_hits.size() << std::endl;
  }
}
