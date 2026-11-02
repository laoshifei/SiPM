#pragma once

#include "G4VModularPhysicsList.hh"

class RunConfig;

class PhysicsList final : public G4VModularPhysicsList {
public:
  explicit PhysicsList(const RunConfig& config);
  void ConstructProcess() override;

private:
  const RunConfig& config_;
};
