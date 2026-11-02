#include "ActionInitialization.hh"
#include "DataStructure.hh"
#include "RootIO.hh"
#include "RunAction.hh"
#include "EventAction.hh"
#include "PrimaryGeneratorAction.hh"
#include "SteppingAction.hh"
#include "StackingAction.hh"
#include "TrackingAction.hh"
#include "SiPMDigitizer.hh"
#include "G4DigiManager.hh"
#include <memory>

ActionInitialization::ActionInitialization(const RunConfig& config) : config_(config) {}

void ActionInitialization::Build() const {
  auto data = std::make_shared<WaveformEventData>();
  auto output = std::make_shared<RootIO>(config_, data);
  G4DigiManager::GetDMpointer()->AddNewModule(new SiPMDigitizer(config_));

  SetUserAction(new PrimaryGeneratorAction(config_));
  SetUserAction(new RunAction(config_, output));
  SetUserAction(new EventAction(config_, data, output));
  SetUserAction(new StackingAction(config_, data));
  SetUserAction(new TrackingAction());
  SetUserAction(new SteppingAction(config_, data));
}
