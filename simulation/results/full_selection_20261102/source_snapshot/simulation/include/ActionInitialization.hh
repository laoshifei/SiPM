#pragma once

#include "G4VUserActionInitialization.hh"

class RunConfig;

class ActionInitialization final : public G4VUserActionInitialization {
public:
  explicit ActionInitialization(const RunConfig& config);
  void Build() const override;

private:
  const RunConfig& config_;
};
