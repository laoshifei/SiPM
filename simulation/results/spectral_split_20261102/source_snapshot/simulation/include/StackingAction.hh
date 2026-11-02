#pragma once

#include "G4UserStackingAction.hh"
#include <memory>

class RunConfig;
struct WaveformEventData;

class StackingAction final : public G4UserStackingAction {
public:
  StackingAction(const RunConfig& config, std::shared_ptr<WaveformEventData> data);
  G4ClassificationOfNewTrack ClassifyNewTrack(const G4Track* track) override;

private:
  const RunConfig& config_;
  std::shared_ptr<WaveformEventData> data_;
};
