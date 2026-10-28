#include "DetectorConstruction.hh"
#include "RunConfig.hh"
#include "Constants.hh"
#include "G4Box.hh"
#include "G4LogicalVolume.hh"
#include "G4PVPlacement.hh"
#include "G4NistManager.hh"
#include "G4Material.hh"
#include "G4MaterialPropertiesTable.hh"
#include "G4OpticalSurface.hh"
#include "G4LogicalSkinSurface.hh"
#include "G4LogicalBorderSurface.hh"
#include "G4UserLimits.hh"
#include "G4VisAttributes.hh"
#include "G4Colour.hh"
#include "G4SystemOfUnits.hh"
#include <algorithm>
#include <stdexcept>
#include <vector>

DetectorConstruction::DetectorConstruction(const RunConfig& config) : config_(config) {}

G4VPhysicalVolume* DetectorConstruction::Construct() {
  auto* nist = G4NistManager::Instance();
  const auto& geometry = config_.Geometry();
  const auto& optical = config_.Optical();

  auto* air = nist->FindOrBuildMaterial(geometry.at("world_material"));
  auto* crystal = nist->FindOrBuildMaterial("G4_CESIUM_IODIDE");
  auto* film = nist->FindOrBuildMaterial("G4_MYLAR");
  auto* teflon = nist->FindOrBuildMaterial("G4_TEFLON");
  // Effective optical media retained from the pilot. Their chemical composition
  // is a placeholder; this refactor does not introduce a detailed SiPM package.
  auto* grease = new G4Material("CouplingGrease", 1.0 * CLHEP::g / cm3, 1);
  grease->AddElement(nist->FindOrBuildElement("C"), 1);
  auto* sensor = new G4Material("VirtualEpoxyEntrance", 1.2 * CLHEP::g / cm3, 1);
  sensor->AddElement(nist->FindOrBuildElement("C"), 1);

  std::vector<double> photon_energy = optical.at("photon_energy_eV");
  std::vector<double> spectrum = optical.at("emission_relative");
  if (photon_energy.size() != spectrum.size() || photon_energy.size() < 2) {
    throw std::runtime_error("Invalid emission spectrum");
  }
  for (auto& energy : photon_energy) energy *= eV;
  auto set_optical_medium = [&](G4Material* material, double index, double absorption_mm) {
    auto* properties = new G4MaterialPropertiesTable;
    std::vector<double> indices(photon_energy.size(), index);
    std::vector<double> lengths(photon_energy.size(), absorption_mm * mm);
    properties->AddProperty("RINDEX", photon_energy.data(), indices.data(), photon_energy.size());
    properties->AddProperty("ABSLENGTH", photon_energy.data(), lengths.data(), photon_energy.size());
    material->SetMaterialPropertiesTable(properties);
  };
  set_optical_medium(air, 1., 100000.);
  set_optical_medium(film, 1.65, .1);
  set_optical_medium(grease, 1.46, 1000.);
  set_optical_medium(sensor, 1.55, 1.);
  set_optical_medium(crystal, optical.at("rindex"), optical.at("absorption_mm"));

  // Tl emission is parameterized in the material properties, not derived from
  // microscopic doping. Fractions are integrated light fractions.
  auto* properties = crystal->GetMaterialPropertiesTable();
  properties->AddProperty("SCINTILLATIONCOMPONENT1", photon_energy.data(), spectrum.data(), photon_energy.size());
  properties->AddProperty("SCINTILLATIONCOMPONENT2", photon_energy.data(), spectrum.data(), photon_energy.size());
  double kinetic_energy[] = {0., .001 * MeV, .01 * MeV, .1 * MeV, 1. * MeV, 10. * MeV, 100. * MeV};
  for (const std::string kind : {"ELECTRON", "ALPHA", "ION", "PROTON", "DEUTERON", "TRITON"}) {
    const bool electron = kind == "ELECTRON";
    const bool alpha = kind == "ALPHA";
    const double ratio = electron ? 1. : (alpha ? optical.at("alpha_light_yield_ratio").get<double>()
                                                : optical.at("ion_light_yield_ratio").get<double>());
    double yields[7];
    for (int i = 0; i < 7; ++i) {
      yields[i] = kinetic_energy[i] / MeV * optical.at("yield_photons_per_MeV").get<double>() * ratio;
    }
    properties->AddProperty((kind + "SCINTILLATIONYIELD").c_str(), kinetic_energy, yields, 7);
    const double fraction = optical.at(electron ? "electron_fast_fraction" : "alpha_fast_fraction");
    const auto times = optical.at(electron ? "electron_tau_ns" : "alpha_tau_ns");
    properties->AddConstProperty((kind + "SCINTILLATIONYIELD1").c_str(), fraction);
    properties->AddConstProperty((kind + "SCINTILLATIONYIELD2").c_str(), 1. - fraction);
    properties->AddConstProperty((kind + "SCINTILLATIONTIMECONSTANT1").c_str(), times.at(0).get<double>() * ns);
    properties->AddConstProperty((kind + "SCINTILLATIONTIMECONSTANT2").c_str(), times.at(1).get<double>() * ns);
  }
  properties->AddConstProperty("SCINTILLATIONTIMECONSTANT1", optical.at("electron_tau_ns").at(0).get<double>() * ns);
  properties->AddConstProperty("SCINTILLATIONTIMECONSTANT2", optical.at("electron_tau_ns").at(1).get<double>() * ns);
  properties->AddConstProperty("RESOLUTIONSCALE", 1.);

  const double world_half = csi::WorldHalfSizeMm * mm;
  auto* world = new G4LogicalVolume(new G4Box("World", world_half, world_half, world_half), air, "World");
  auto* world_physical = new G4PVPlacement(nullptr, {}, world, "World", nullptr, false, 0, true);
  const auto dimensions = geometry.at("crystal_mm");
  const double x = dimensions.at(0).get<double>() * mm / 2;
  const double y = dimensions.at(1).get<double>() * mm / 2;
  const double z = dimensions.at(2).get<double>() * mm / 2;
  auto* crystal_logical = new G4LogicalVolume(new G4Box("CsI", x, y, z), crystal, "CsI");
  auto* crystal_physical = new G4PVPlacement(nullptr, {}, crystal_logical, "CsI", world, false, 0, true);
  crystal_logical->SetUserLimits(new G4UserLimits(optical.at("charged_step_um").get<double>() * um));

  const double thickness = geometry.at("mylar_um").get<double>() * um;
  auto* mylar = new G4LogicalVolume(new G4Box("Mylar", x, y, thickness / 2), film, "Mylar");
  auto* film_physical = new G4PVPlacement(nullptr, {0, 0, -z - thickness / 2}, mylar, "Mylar", world, false, 0, true);
  mylar->SetUserLimits(new G4UserLimits(std::min(thickness / 5, 1. * um)));

  const double coupling = geometry.at("grease_mm").get<double>() * mm;
  auto* grease_logical = new G4LogicalVolume(new G4Box("Grease", x, y, coupling / 2), grease, "Grease");
  auto* grease_physical = new G4PVPlacement(nullptr, {0, 0, z + coupling / 2}, grease_logical, "Grease", world, false, 0, true);
  const double sensor_x = std::max(x, geometry.at("array_nx").get<double>() * geometry.at("channel_pitch_mm").get<double>() * mm / 2);
  const double sensor_y = std::max(y, geometry.at("array_ny").get<double>() * geometry.at("channel_pitch_mm").get<double>() * mm / 2);
  auto* sensor_logical = new G4LogicalVolume(new G4Box("Sensor", sensor_x, sensor_y, .05 * mm), sensor, "Sensor");
  new G4PVPlacement(nullptr, {0, 0, z + coupling + .05 * mm}, sensor_logical, "Sensor", world, false, 0, true);

  auto* wrap = new G4OpticalSurface("PTFE effective surface", unified, ground, dielectric_metal);
  auto* wrap_properties = new G4MaterialPropertiesTable;
  std::vector<double> reflectivity(photon_energy.size(), optical.at("ptfe_reflectivity"));
  wrap_properties->AddProperty("REFLECTIVITY", photon_energy.data(), reflectivity.data(), photon_energy.size());
  wrap->SetMaterialPropertiesTable(wrap_properties);
  new G4LogicalSkinSurface("PTFE on lateral faces", crystal_logical, wrap);

  // Explicit front/back borders override the effective side skin.
  auto* clear = new G4OpticalSurface("Uncoated interfaces", unified, polished, dielectric_dielectric);
  new G4LogicalBorderSurface("CsI to Mylar", crystal_physical, film_physical, clear);
  new G4LogicalBorderSurface("Mylar to CsI", film_physical, crystal_physical, clear);
  new G4LogicalBorderSurface("CsI to grease", crystal_physical, grease_physical, clear);
  new G4LogicalBorderSurface("Grease to CsI", grease_physical, crystal_physical, clear);

  // Physical PTFE sheets are used for charged-particle transport. Their optical
  // response is the effective side surface above, as in the pilot model.
  const double tape = geometry.at("ptfe_side_mm").get<double>() * mm;
  for (int axis = 0; axis < 2; ++axis) {
    for (int sign : {-1, 1}) {
      auto* solid = new G4Box("PTFE", axis == 0 ? tape / 2 : x, axis == 1 ? tape / 2 : y + tape, z);
      auto* logical = new G4LogicalVolume(solid, teflon, "PTFE");
      const auto position = axis == 0 ? G4ThreeVector(sign * (x + tape / 2), 0, 0)
                                      : G4ThreeVector(0, sign * (y + tape / 2), 0);
      new G4PVPlacement(nullptr, position, logical, "PTFE", world, false, axis * 2 + (sign + 1) / 2, true);
      logical->SetVisAttributes(G4VisAttributes(G4Colour(.85, .85, .85, .35)));
    }
  }

  world->SetVisAttributes(G4VisAttributes::GetInvisible());
  crystal_logical->SetVisAttributes(G4VisAttributes(G4Colour(.25, .75, .3, .55)));
  mylar->SetVisAttributes(G4VisAttributes(G4Colour(.25, .45, .9, .7)));
  grease_logical->SetVisAttributes(G4VisAttributes(G4Colour(.7, .85, 1., .35)));
  sensor_logical->SetVisAttributes(G4VisAttributes(G4Colour(.8, .25, .25, .8)));
  return world_physical;
}
