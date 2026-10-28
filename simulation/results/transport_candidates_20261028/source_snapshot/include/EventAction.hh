#pragma once

#include "G4UserEventAction.hh"
#include <memory>

class RunConfig;
class RootIO;
struct WaveformEventData;

class EventAction final : public G4UserEventAction {
public:
  EventAction(const RunConfig& config, std::shared_ptr<WaveformEventData> data,
              std::shared_ptr<RootIO> output);
  void BeginOfEventAction(const G4Event* event) override;
  void EndOfEventAction(const G4Event* event) override;

private:
  const RunConfig& config_;
  std::shared_ptr<WaveformEventData> data_;
  std::shared_ptr<RootIO> output_;
};
