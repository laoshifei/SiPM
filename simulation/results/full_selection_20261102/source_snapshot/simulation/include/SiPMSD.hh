#pragma once

#include "G4VSensitiveDetector.hh"
#include "SiPMHit.hh"
#include <vector>

class RunConfig;
class G4OpBoundaryProcess;

// Entrance-surface scorer: SteppingAction forwards the step reaching Sensor to
// this SD. This avoids advancing the photon one artificial step inside epoxy.
// The current entry convention is audited through boundary_status on each hit.
class SiPMSD final : public G4VSensitiveDetector {
public:
  static constexpr const char* Name = "SiPM";
  static constexpr const char* Collection = "SiPM/OpticalArrivals";
  explicit SiPMSD(const RunConfig& config);
  void Initialize(G4HCofThisEvent* collections) override;
  G4bool ProcessHits(G4Step* step, G4TouchableHistory*) override;

private:
  SiPMHitsCollection* hits_ = nullptr;  // Owned by this event's G4HCofThisEvent.
  int collection_id_ = -1;
  int nx_;
  int ny_;
  double pitch_mm_;
  double active_mm_;
  std::vector<int> summed_groups_;
  G4OpBoundaryProcess* boundary_ = nullptr;
};
