#include "SteppingAction.hh"
#include "RunConfig.hh"
#include "DataStructure.hh"
#include "Constants.hh"
#include "TrackInformation.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4SystemOfUnits.hh"
#include <algorithm>
#include <cmath>
#include <utility>

SteppingAction::SteppingAction(const RunConfig& config, std::shared_ptr<WaveformEventData> data)
    : config_(config), data_(std::move(data)) {}

void SteppingAction::UserSteppingAction(const G4Step* step) {
  auto* track = step->GetTrack();
  const auto* pre = step->GetPreStepPoint();
  const auto* post = step->GetPostStepPoint();
  if (post->GetGlobalTime() / ns >= config_.DurationNs()) {
    track->SetTrackStatus(fStopAndKill);
    return;
  }
  const auto* volume = pre->GetPhysicalVolume();
  if (!volume) return;
  const auto* information = dynamic_cast<const TrackInformation*>(track->GetUserInformation());
  const int origin = information ? static_cast<int>(information->origin) : 0;

  if (track->GetDefinition() == G4OpticalPhoton::Definition()) {
    if (post->GetPhysicalVolume() && post->GetPhysicalVolume()->GetName() == "Sensor") {
      const auto& geometry = config_.Geometry();
      const auto position = post->GetPosition();
      const double x = position.x() / mm;
      const double y = position.y() / mm;
      const double pitch = geometry.at("channel_pitch_mm");
      const double width = geometry.at("channel_active_mm");
      const int nx = geometry.at("array_nx");
      const int ny = geometry.at("array_ny");
      const int ix = int(std::floor(x / pitch + nx / 2.));
      const int iy = int(std::floor(y / pitch + ny / 2.));
      if (ix >= 0 && iy >= 0 && ix < nx && iy < ny) {
        const double local_x = x - (ix - (nx - 1) / 2.) * pitch;
        const double local_y = y - (iy - (ny - 1) / 2.) * pitch;
        const auto channels = geometry.at("summed_channels").get<std::vector<int>>();
        const auto selected = std::find(channels.begin(), channels.end(), iy * nx + ix);
        if (std::abs(local_x) < width / 2 && std::abs(local_y) < width / 2 && selected != channels.end()) {
          const int side = csi::EffectiveGridSide;
          const int gx = std::clamp(int((local_x / width + .5) * side), 0, side - 1);
          const int gy = std::clamp(int((local_y / width + .5) * side), 0, side - 1);
          const int cell = int((gy * side + gx) * double(csi::CellsPerChannel) / (side * side));
          const int group = selected - channels.begin();
          const double time = post->GetGlobalTime() / ns;
          const double energy = track->GetKineticEnergy() / eV;
          data_->photon_hits.push_back({time, x, y, energy, group, cell});
          data_->photon_t_ns.push_back(time);
          data_->photon_x_mm.push_back(x);
          data_->photon_y_mm.push_back(y);
          data_->photon_e_eV.push_back(energy);
          data_->photon_origin.push_back(origin);
          data_->photon_emitter_pdg.push_back(information ? information->optical_emitter_pdg : 0);
        }
      }
      // Count an entrance once. Semiconductor transport/direct charge is outside
      // this effective optical readout model; no extra sensor step is introduced.
      track->SetTrackStatus(fStopAndKill);
    }
    return;
  }

  const double energy = step->GetTotalEnergyDeposit() / MeV;
  if (volume->GetName() == "CsI") {
    data_->edep_MeV += energy;
    data_->edep_by_origin_MeV.at(origin) += energy;
    const int pdg = track->GetDefinition()->GetPDGEncoding();
    if (pdg == 1000020040) {
      data_->edep_alpha_MeV += energy;
    } else if (std::abs(pdg) == 11) {
      data_->edep_electron_MeV += energy;
    } else if (track->GetDefinition()->GetParticleType() == "nucleus") {
      data_->edep_ion_MeV += energy;
    }
  } else if (volume->GetName() == "Mylar") {
    data_->edep_mylar_MeV += energy;
  } else if (volume->GetName() == "World") {
    data_->edep_air_MeV += energy;
  }
}
