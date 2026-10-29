#pragma once

#include "G4VUserDetectorConstruction.hh"
#include <memory>

class RunConfig;
class G4GenericMessenger;

class DetectorConstruction final : public G4VUserDetectorConstruction {
public:
  explicit DetectorConstruction(const RunConfig& config);
  ~DetectorConstruction() override;
  G4VPhysicalVolume* Construct() override;
  void ConstructSDandField() override;
  void ExportGeometry(G4String filename);

private:
  const RunConfig& config_;
  G4VPhysicalVolume* world_ = nullptr;
  std::unique_ptr<G4GenericMessenger> messenger_;
};
