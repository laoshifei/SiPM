#include "SiPMSD.hh"
#include "RunConfig.hh"
#include "Constants.hh"
#include "TrackInformation.hh"
#include "G4Step.hh"
#include "G4Track.hh"
#include "G4OpticalPhoton.hh"
#include "G4OpBoundaryProcess.hh"
#include "G4ProcessManager.hh"
#include "G4ProcessVector.hh"
#include "G4HCofThisEvent.hh"
#include "G4SystemOfUnits.hh"
#include <algorithm>
#include <cmath>
#include <stdexcept>

SiPMSD::SiPMSD(const RunConfig& config) : G4VSensitiveDetector(Name) {
  collectionName.push_back("OpticalArrivals");
  const auto& geometry = config.Geometry();
  nx_ = geometry.at("array_nx");
  ny_ = geometry.at("array_ny");
  pitch_mm_ = geometry.at("channel_pitch_mm");
  active_mm_ = geometry.at("channel_active_mm");
  summed_groups_.assign(nx_ * ny_, -1);
  const auto channels = geometry.at("summed_channels").get<std::vector<int>>();
  for (std::size_t group = 0; group < channels.size(); ++group) summed_groups_.at(channels[group]) = group;
}

void SiPMSD::Initialize(G4HCofThisEvent* collections) {
  hits_ = new SiPMHitsCollection(SensitiveDetectorName, collectionName[0]);
  if (collection_id_ < 0) collection_id_ = GetCollectionID(0);
  collections->AddHitsCollection(collection_id_, hits_);
}

G4bool SiPMSD::ProcessHits(G4Step* step, G4TouchableHistory*) {
  auto* track = step->GetTrack();
  if (track->GetDefinition() != G4OpticalPhoton::Definition()) return false;
  const auto* post = step->GetPostStepPoint();
  const auto* pre = step->GetPreStepPoint();
  if (!post->GetPhysicalVolume() || post->GetPhysicalVolume()->GetName() != "Sensor"
      || (pre->GetPhysicalVolume() && pre->GetPhysicalVolume()->GetName() == "Sensor")
      || post->GetStepStatus() != fGeomBoundary) return false;
  if (!hits_) throw std::runtime_error("SiPM hit collection was not initialized");
  if (!boundary_) {
    const auto* processes = track->GetDefinition()->GetProcessManager()->GetProcessList();
    for (std::size_t i = 0; i < processes->size(); ++i) {
      if (auto* process = dynamic_cast<G4OpBoundaryProcess*>((*processes)[i])) boundary_ = process;
    }
    if (!boundary_) throw std::runtime_error("Optical boundary process is missing");
  }
  const auto status = boundary_->GetStatus();
  // Post-step touchable can name Sensor even after Fresnel reflection. Only
  // transmitted photons reach the effective entrance; reflected tracks remain
  // alive and may return later. PDE is applied once, by the digitizer.
  if (status != Transmission && status != FresnelRefraction && status != SameMaterial) return false;
  track->SetTrackStatus(fStopAndKill);

  const auto position = post->GetPosition();
  const double x = position.x() / mm;
  const double y = position.y() / mm;
  const int ix = int(std::floor(x / pitch_mm_ + nx_ / 2.));
  const int iy = int(std::floor(y / pitch_mm_ + ny_ / 2.));
  if (ix < 0 || iy < 0 || ix >= nx_ || iy >= ny_) return false;
  const double local_x = x - (ix - (nx_ - 1) / 2.) * pitch_mm_;
  const double local_y = y - (iy - (ny_ - 1) / 2.) * pitch_mm_;
  const int group = summed_groups_[iy * nx_ + ix];
  if (std::abs(local_x) >= active_mm_ / 2 || std::abs(local_y) >= active_mm_ / 2 || group < 0) return false;

  const int side = csi::EffectiveGridSide;
  const int gx = std::clamp(int((local_x / active_mm_ + .5) * side), 0, side - 1);
  const int gy = std::clamp(int((local_y / active_mm_ + .5) * side), 0, side - 1);
  const int cell = int((gy * side + gx) * double(csi::CellsPerChannel) / (side * side));
  const auto* information = dynamic_cast<const TrackInformation*>(track->GetUserInformation());
  auto* hit = new SiPMHit;
  hit->photon = {post->GetGlobalTime() / ns, x, y, track->GetKineticEnergy() / eV, group, cell};
  hit->origin = information ? static_cast<int>(information->origin) : 0;
  hit->emitter_pdg = information ? information->optical_emitter_pdg : 0;
  hit->boundary_status = status;
  hits_->insert(hit);
  return true;
}
