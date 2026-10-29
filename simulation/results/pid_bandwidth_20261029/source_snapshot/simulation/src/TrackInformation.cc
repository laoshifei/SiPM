#include "TrackInformation.hh"
#include "G4ios.hh"

void TrackInformation::Print() const {
  G4cout << "Decay origin=" << static_cast<int>(origin)
         << ", decay particle PDG=" << decay_particle_pdg
         << ", optical emitter PDG=" << optical_emitter_pdg << G4endl;
}
