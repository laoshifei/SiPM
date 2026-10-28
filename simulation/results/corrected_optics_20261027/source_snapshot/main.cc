#include "DetectorConstruction.hh"
#include "PhysicsList.hh"
#include "ActionInitialization.hh"
#include "RunConfig.hh"
#include "G4RunManagerFactory.hh"
#include "G4UImanager.hh"
#include "G4UIExecutive.hh"
#include "G4VisExecutive.hh"
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

int main(int argc, char** argv) {
  try {
    if (argc != 1 && argc != 2 && argc != 6) {
      throw std::runtime_error(
          "Usage: csi_sources [run.mac] OR CONFIG.json Am241|Co60 EVENTS SEED OUTPUT.root");
    }
    RunConfig config(std::string(CSI_SOURCE_DIR) + "/configs/csi_tl_initial.json");
    const bool legacy_cli = argc == 6;
    if (legacy_cli) {
      config.Load(argv[1]);
      config.SetSource(argv[2]);
      config.SetExpectedEvents(std::stoi(argv[3]));
      config.SetSeed(std::stol(argv[4]));
      config.SetOutput(argv[5]);
    }

    // Serial by design: one isolated parent decay and its ADC record per event.
    std::unique_ptr<G4RunManager> run_manager(
        G4RunManagerFactory::CreateRunManager(G4RunManagerType::SerialOnly));
    run_manager->SetUserInitialization(new DetectorConstruction(config));
    run_manager->SetUserInitialization(new PhysicsList());
    run_manager->SetUserInitialization(new ActionInitialization(config));

    if (legacy_cli) {
      run_manager->Initialize();
      run_manager->BeamOn(config.ExpectedEvents());
      return 0;
    }

    // Macro/interactive entry points match the reference GAGG project. Loading
    // visualization drivers does not open a window; /vis/open controls that.
    auto visualization = std::make_unique<G4VisExecutive>("quiet");
    visualization->Initialize();
    auto* commands = G4UImanager::GetUIpointer();
    if (argc == 2) {
      return commands->ApplyCommand(G4String("/control/execute ") + argv[1]) == 0 ? 0 : 1;
    }
    auto ui = std::make_unique<G4UIExecutive>(argc, argv);
    const auto status = commands->ApplyCommand(
        G4String("/control/execute ") + CSI_SOURCE_DIR + "/macros/init_vis.mac");
    if (status != 0) return 1;
    ui->SessionStart();
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "ERROR: " << error.what() << '\n';
    return 1;
  }
}
