#pragma once

#include "G4VUserPrimaryGeneratorAction.hh"
#include "G4ParticleGun.hh"

class RunConfig;

class PrimaryGeneratorAction final : public G4VUserPrimaryGeneratorAction {
public:
  explicit PrimaryGeneratorAction(const RunConfig& config);
  void GeneratePrimaries(G4Event* event) override;

private:
  const RunConfig& config_;
  G4ParticleGun gun_{1};
};
