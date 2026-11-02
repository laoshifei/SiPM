#include "TrackingAction.hh"
#include "TrackInformation.hh"
#include "G4Track.hh"
#include "G4TrackingManager.hh"
#include "G4OpticalPhoton.hh"
#include "G4VProcess.hh"
#include <cmath>
#include <stdexcept>

namespace {
DecayOrigin OriginOfDecayProduct(const G4Track* track) {
  const int pdg = track->GetDefinition()->GetPDGEncoding();
  if (pdg == 1000020040) return DecayOrigin::Alpha;
  if (pdg == 22) return DecayOrigin::Gamma;
  if (std::abs(pdg) == 11) return DecayOrigin::DecayElectron;
  if (track->GetDefinition()->GetParticleType() == "nucleus") return DecayOrigin::Recoil;
  return DecayOrigin::Other;
}
}  // namespace

void TrackingAction::PreUserTrackingAction(const G4Track* track) {
  if (track->GetUserInformation()) {
    if (!dynamic_cast<TrackInformation*>(track->GetUserInformation())) {
      throw std::runtime_error("Unexpected user track information type");
    }
    return;
  }
  // The source ion initially has no radioactive-product ancestor. Secondary
  // information is attached in the parent's PostUserTrackingAction below.
  const_cast<G4Track*>(track)->SetUserInformation(new TrackInformation);
}

void TrackingAction::PostUserTrackingAction(const G4Track* track) {
  const auto* parent = dynamic_cast<const TrackInformation*>(track->GetUserInformation());
  if (!parent) throw std::runtime_error("Missing parent ancestry information");
  const auto* secondaries = fpTrackingManager->GimmeSecondaries();
  if (!secondaries) return;

  for (auto* secondary : *secondaries) {
    if (!secondary || secondary->GetUserInformation()) continue;
    auto* information = new TrackInformation(*parent);
    const auto* creator = secondary->GetCreatorProcess();
    if (creator && creator->GetProcessName() == "RadioactiveDecay") {
      // A prompt gamma from an excited daughter starts its own gamma ancestry;
      // do not inherit the recoil identity of that daughter nucleus.
      information->origin = OriginOfDecayProduct(secondary);
      information->decay_particle_pdg = secondary->GetDefinition()->GetPDGEncoding();
    }
    if (secondary->GetDefinition() == G4OpticalPhoton::Definition() && creator
        && (creator->GetProcessName() == "Scintillation" || creator->GetProcessName() == "Cerenkov")) {
      information->optical_emitter_pdg = track->GetDefinition()->GetPDGEncoding();
    }
    secondary->SetUserInformation(information);  // Owned by the G4Track.
  }
}
