#include "PhysicsList.hh"
#include "RunConfig.hh"
#include "G4EmLivermorePhysics.hh"
#include "G4DecayPhysics.hh"
#include "G4RadioactiveDecayPhysics.hh"
#include "G4OpticalPhysics.hh"
#include "G4OpticalParameters.hh"
#include "G4StepLimiterPhysics.hh"
#include "G4HadronicParameters.hh"
#include "G4SystemOfUnits.hh"

PhysicsList::PhysicsList(const RunConfig& config) : config_(config) {
  SetVerboseLevel(0);
  defaultCutValue = .1 * um;
  G4HadronicParameters::Instance()->SetTimeThresholdForRadioactiveDecay(1e30 * year);
  RegisterPhysics(new G4EmLivermorePhysics(0));
  RegisterPhysics(new G4DecayPhysics(0));
  RegisterPhysics(new G4RadioactiveDecayPhysics(0));
  RegisterPhysics(new G4OpticalPhysics(0));
  RegisterPhysics(new G4StepLimiterPhysics());
  G4OpticalParameters::Instance()->SetScintByParticleType(true);
  G4OpticalParameters::Instance()->SetScintTrackSecondariesFirst(false);
}

void PhysicsList::ConstructProcess() {
  // Read after any PreInit macro configuration, before G4Scintillation is built.
  const auto rise = config_.Optical().value("rise_time_ns", std::vector<double>{0., 0.});
  G4OpticalParameters::Instance()->SetScintFiniteRiseTime(rise[0] > 0. || rise[1] > 0.);
  G4VModularPhysicsList::ConstructProcess();
}
