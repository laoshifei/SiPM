#include "SiPMDigitizer.hh"
#include "RunConfig.hh"
#include "DataStructure.hh"
#include "Constants.hh"
#include <algorithm>
#include <memory>
#include <numeric>
#include <stdexcept>

SiPMDigitizer::SiPMDigitizer(const RunConfig& config)
    : G4VDigitizerModule(Name), config_(config) {
  collectionName.push_back("SiPMWaveforms");
}

void SiPMDigitizer::SetInput(const WaveformEventData& event) {
  input_ = &event;
}

void SiPMDigitizer::Digitize() {
  if (!input_) throw std::runtime_error("SiPMDigitizer has no current event input");
  const auto* event = input_;
  input_ = nullptr;
  std::vector<sipm::Photon> photons;
  photons.reserve(event->photon_hits.size());
  std::vector<std::size_t> order(event->photon_hits.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
    return event->photon_hits[a].time_ns < event->photon_hits[b].time_ns;
  });
  for (auto index : order) {
    const auto& hit = event->photon_hits[index];
    // Adapter for the unchanged GAGG numerical backend. Its x/y fields index
    // effective cells here, not positions in the Geant4 detector geometry.
    const double cell_coordinate = hit.summed_channel_index * csi::CellsPerChannel + hit.microcell_index + .5;
    photons.push_back({hit.time_ns, cell_coordinate, .5, hit.energy_eV});
  }
  auto digi = std::make_unique<SiPMDigi>();
  digi->response = sipm::Simulate(config_.SensorModel(), photons, config_.DurationNs(),
      config_.PretriggerNs(), config_.DurationNs(),
      sipm::EventSeed(config_.Seed() + 17, event->event_id),
      sipm::EventSeed(config_.Seed() + 31, event->event_id));
  digi->response.event_id = event->event_id;
  // Attribution follows the realized avalanches after PDE, recovery and any
  // secondary avalanches. It is diagnostic only and never changes probability.
  for (std::size_t i = 0; i < digi->response.avalanche_time_ns.size(); ++i) {
    const int photon = digi->response.avalanche_photon_index[i];
    const int parent = digi->response.avalanche_parent[i];
    int origin = 0;
    int emitter = 0;
    if (photon >= 0) {
      const auto original_index = order.at(photon);
      origin = event->photon_origin.at(original_index);
      emitter = event->photon_emitter_pdg.at(original_index);
    } else if (parent >= 0) {
      origin = digi->avalanche_origin.at(parent);
      emitter = digi->avalanche_emitter_pdg.at(parent);
    }
    digi->avalanche_origin.push_back(origin);
    digi->avalanche_emitter_pdg.push_back(emitter);
  }
  auto collection = std::make_unique<SiPMDigiCollection>(moduleName, collectionName[0]);
  collection->insert(digi.release());
  StoreDigiCollection(collection.release());  // This G4Event owns the collection.
}
