#include "SiPMHit.hh"
#include "G4ios.hh"

void SiPMHit::Print() {
  G4cout << "SiPM optical entry: t=" << photon.time_ns << " ns, x=" << photon.x_mm
         << " mm, y=" << photon.y_mm << " mm, E=" << photon.energy_eV
         << " eV, origin=" << origin << ", boundary status=" << boundary_status << G4endl;
}
