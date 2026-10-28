#pragma once

#include "G4VUserDetectorConstruction.hh"

class RunConfig;

class DetectorConstruction final : public G4VUserDetectorConstruction {
public:
  explicit DetectorConstruction(const RunConfig& config);
  G4VPhysicalVolume* Construct() override;

private:
  const RunConfig& config_;
};
