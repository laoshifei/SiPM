#pragma once

#include "G4VUserTrackInformation.hh"
#include "globals.hh"

// These are simulation ancestry categories, not experimental particle labels.
// "DecayElectron" includes beta, conversion and atomic-relaxation electrons;
// it must not be reported as a pure beta category without further information.
enum class DecayOrigin : int {
  Unknown = 0,
  Alpha = 1,
  Gamma = 2,
  DecayElectron = 3,
  Recoil = 4,
  Other = 5
};
constexpr int DecayOriginCount = 6;

class TrackInformation final : public G4VUserTrackInformation {
public:
  TrackInformation() = default;
  TrackInformation(const TrackInformation&) = default;
  void Print() const override;

  DecayOrigin origin = DecayOrigin::Unknown;
  G4int decay_particle_pdg = 0;
  G4int optical_emitter_pdg = 0;
};
