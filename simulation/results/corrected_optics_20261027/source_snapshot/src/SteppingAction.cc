#include "SteppingAction.hh"
#include "RunConfig.hh"
#include "DataStructure.hh"
#include "TrackInformation.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4SystemOfUnits.hh"
#include <cmath>
#include <stdexcept>
#include "G4VSensitiveDetector.hh"
#include <utility>

SteppingAction::SteppingAction(const RunConfig& config, std::shared_ptr<WaveformEventData> data)
    : config_(config), data_(std::move(data)) {}

void SteppingAction::UserSteppingAction(const G4Step* step) {
  auto* track = step->GetTrack();
  const auto* pre = step->GetPreStepPoint();
  const auto* post = step->GetPostStepPoint();
  if (post->GetGlobalTime() / ns >= config_.DurationNs()) {
    track->SetTrackStatus(fStopAndKill);
    return;
  }
  const auto* volume = pre->GetPhysicalVolume();
  if (!volume) return;
  const auto* information = dynamic_cast<const TrackInformation*>(track->GetUserInformation());
  const int origin = information ? static_cast<int>(information->origin) : 0;

  if (track->GetDefinition() == G4OpticalPhoton::Definition()) {
    if (post->GetPhysicalVolume() && post->GetPhysicalVolume()->GetName() == "Sensor") {
      auto* sensor = post->GetSensitiveDetector();
      if (!sensor) throw std::runtime_error("Sensor sensitive detector is not attached");
      sensor->Hit(const_cast<G4Step*>(step));
      // SiPMSD consumes transmitted photons only. Reflected photons continue
      // through Geant4 and must not be counted or terminated at this boundary.
    }
    return;
  }

  const double energy = step->GetTotalEnergyDeposit() / MeV;
  if (volume->GetName() == "CsI") {
    data_->edep_MeV += energy;
    data_->edep_by_origin_MeV.at(origin) += energy;
    const int pdg = track->GetDefinition()->GetPDGEncoding();
    if (pdg == 1000020040) {
      data_->edep_alpha_MeV += energy;
    } else if (std::abs(pdg) == 11) {
      data_->edep_electron_MeV += energy;
    } else if (track->GetDefinition()->GetParticleType() == "nucleus") {
      data_->edep_ion_MeV += energy;
    }
  } else if (volume->GetName() == "Mylar") {
    data_->edep_mylar_MeV += energy;
  } else if (volume->GetName() == "World") {
    data_->edep_air_MeV += energy;
  }
}
