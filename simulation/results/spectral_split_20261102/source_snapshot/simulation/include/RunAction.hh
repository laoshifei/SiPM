#pragma once

#include "G4UserRunAction.hh"
#include <memory>

class RunConfig;
class RootIO;

class RunAction final : public G4UserRunAction {
public:
  RunAction(const RunConfig& config, std::shared_ptr<RootIO> output);
  void BeginOfRunAction(const G4Run* run) override;
  void EndOfRunAction(const G4Run* run) override;

private:
  const RunConfig& config_;
  std::shared_ptr<RootIO> output_;
};
