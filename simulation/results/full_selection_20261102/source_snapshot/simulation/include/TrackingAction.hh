#pragma once

#include "G4UserTrackingAction.hh"

class TrackingAction final : public G4UserTrackingAction {
public:
  void PreUserTrackingAction(const G4Track* track) override;
  void PostUserTrackingAction(const G4Track* track) override;
};
