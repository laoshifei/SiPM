#include "StackingAction.hh"
#include "RunConfig.hh"
#include "DataStructure.hh"
#include "TrackInformation.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4VProcess.hh"
#include "G4SystemOfUnits.hh"
#include <utility>

StackingAction::StackingAction(const RunConfig& config, std::shared_ptr<WaveformEventData> data)
    : config_(config), data_(std::move(data)) {}

G4ClassificationOfNewTrack StackingAction::ClassifyNewTrack(const G4Track* track) {
  const bool optical = track->GetDefinition() == G4OpticalPhoton::Definition();
  const auto* creator = track->GetCreatorProcess();
  if (optical) {
    ++data_->n_optical_created;
    if (creator && creator->GetProcessName() == "Scintillation") {
      ++data_->n_scintillation;
      const auto* information = dynamic_cast<const TrackInformation*>(track->GetUserInformation());
      const int origin = information ? static_cast<int>(information->origin) : 0;
      ++data_->scintillation_by_origin.at(origin);
    }
  }
  if (track->GetGlobalTime() / ns >= config_.DurationNs()) {
    if (optical) ++data_->n_late;
    return fKill;
  }
  // Include prompt daughter de-excitation, not only ParentID == 1.
  if (!optical && creator && creator->GetProcessName() == "RadioactiveDecay") {
    data_->decay_pdg.push_back(track->GetDefinition()->GetPDGEncoding());
    data_->decay_parent_id.push_back(track->GetParentID());
    data_->decay_energy_MeV.push_back(track->GetKineticEnergy() / MeV);
    data_->decay_time_ns.push_back(track->GetGlobalTime() / ns);
  }
  return fUrgent;
}
