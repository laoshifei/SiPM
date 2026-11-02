#pragma once

#include "G4UserSteppingAction.hh"
#include <memory>

class RunConfig;
struct WaveformEventData;

// Scores optical arrivals at the sensor entrance, before any step inside it.
// Retaining boundary scoring preserves the established transport RNG sequence.
class SteppingAction final : public G4UserSteppingAction {
public:
  SteppingAction(const RunConfig& config, std::shared_ptr<WaveformEventData> data);
  void UserSteppingAction(const G4Step* step) override;

private:
  const RunConfig& config_;
  std::shared_ptr<WaveformEventData> data_;
};
