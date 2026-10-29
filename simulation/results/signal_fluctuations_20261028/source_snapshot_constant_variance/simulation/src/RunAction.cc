#include "RunAction.hh"
#include "RunConfig.hh"
#include "RootIO.hh"
#include "G4Run.hh"
#include "Randomize.hh"
#include <stdexcept>
#include <utility>

RunAction::RunAction(const RunConfig& config, std::shared_ptr<RootIO> output)
    : config_(config), output_(std::move(output)) {}

void RunAction::BeginOfRunAction(const G4Run* run) {
  const int events = run->GetNumberOfEventToBeProcessed();
  if (config_.ExpectedEvents() && config_.ExpectedEvents() != events) {
    throw std::runtime_error("/run/beamOn count differs from the CLI requested count");
  }
  if (CLHEP::HepRandom::getTheSeed() != static_cast<long>(config_.Seed())) {
    throw std::runtime_error("Use /csi/seed before initialization to set transport and SiPM seeds together");
  }
  output_->Open(events);
}

void RunAction::EndOfRunAction(const G4Run* run) {
  output_->Close(run->GetNumberOfEvent());
}
