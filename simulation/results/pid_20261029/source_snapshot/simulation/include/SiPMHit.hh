#pragma once

#include "DataStructure.hh"
#include "G4VHit.hh"
#include "G4THitsCollection.hh"

class SiPMHit final : public G4VHit {
public:
  OpticalPhotonHitData photon{};
  int origin = 0;
  int emitter_pdg = 0;
  int boundary_status = 0;
  void Print() override;
};

using SiPMHitsCollection = G4THitsCollection<SiPMHit>;
