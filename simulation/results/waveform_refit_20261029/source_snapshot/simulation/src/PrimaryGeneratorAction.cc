#include "PrimaryGeneratorAction.hh"
#include "RunConfig.hh"
#include "G4Event.hh"
#include "G4IonTable.hh"
#include "G4ParticleDefinition.hh"
#include "G4PhysicalConstants.hh"
#include "G4SystemOfUnits.hh"
#include "Randomize.hh"
#include <cmath>

PrimaryGeneratorAction::PrimaryGeneratorAction(const RunConfig& config) : config_(config) {}

void PrimaryGeneratorAction::GeneratePrimaries(G4Event* event) {
  const bool americium = config_.Source() == "Am241";
  auto* ion = G4IonTable::GetIonTable()->GetIon(americium ? 95 : 27, americium ? 241 : 60, 0.);
  // Condition on an isolated parent decay now. Daughter lifetimes/branches stay
  // unchanged; avoid subtracting a centuries-long time with lost ns precision.
  ion->SetPDGStable(false);
  ion->SetPDGLifeTime(0.);
  gun_.SetParticleDefinition(ion);
  gun_.SetParticleEnergy(0.);
  gun_.SetParticleCharge(0.);

  const auto& geometry = config_.Geometry();
  const double radius = geometry.at("source_radius_mm").get<double>() * std::sqrt(G4UniformRand());
  const double phi = twopi * G4UniformRand();
  const double z = -geometry.at("crystal_mm").at(2).get<double>() / 2
      - geometry.at("mylar_um").get<double>() * .001 - geometry.at("source_gap_mm").get<double>();
  gun_.SetParticlePosition({radius * std::cos(phi) * mm, radius * std::sin(phi) * mm, z * mm});
  gun_.SetParticleMomentumDirection({0, 0, 1});
  gun_.SetParticleTime(config_.PretriggerNs() * ns);
  gun_.GeneratePrimaryVertex(event);
}
