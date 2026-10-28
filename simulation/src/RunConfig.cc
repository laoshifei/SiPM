#include "RunConfig.hh"
#include "Constants.hh"
#include "G4GenericMessenger.hh"
#include "G4ApplicationState.hh"
#include "Randomize.hh"
#include <fstream>
#include <set>
#include <stdexcept>

namespace {
void Require(bool condition, const std::string& message) {
  if (!condition) throw std::runtime_error(message);
}
}  // namespace

RunConfig::RunConfig(const std::string& filename) {
  Load(filename);
  SetSeed(1);
  messenger_ = std::make_unique<G4GenericMessenger>(this, "/csi/", "CsI source and readout configuration");
  messenger_->DeclareMethod("config", &RunConfig::Load, "Load geometry/optical/SiPM JSON before initialization")
      .SetStates(G4State_PreInit);
  messenger_->DeclareMethod("source", &RunConfig::SetSource, "Am241 or Co60")
      .SetStates(G4State_PreInit);
  messenger_->DeclareMethod("seed", &RunConfig::SetSeed, "Common transport and electronics base seed")
      .SetStates(G4State_PreInit);
  messenger_->DeclareMethod("output", &RunConfig::SetOutput, "New ROOT output path; never overwrite")
      .SetStates(G4State_PreInit);
}

RunConfig::~RunConfig() = default;

void RunConfig::Load(G4String filename) {
  std::ifstream input(filename);
  Require(bool(input), "Cannot read configuration: " + filename);
  nlohmann::json candidate;
  input >> candidate;
  const auto& g = candidate.at("geometry");
  const auto& o = candidate.at("optical");
  const auto& a = candidate.at("acquisition");
  Require(a.value("digitize", true) || o.at("yield_photons_per_MeV").get<double>() == 0.,
          "Disabling digitization is supported only for zero-light transport scans");

  Require(g.at("material") == "CsI(Tl)", "Only CsI(Tl) is implemented");
  Require(g.at("crystal_mm").size() == 3, "Expected three crystal dimensions");
  for (double dimension : g.at("crystal_mm")) {
    Require(dimension > 0 && dimension < 150, "Invalid crystal dimension");
  }
  Require(g.at("source_gap_mm").get<double>() > 0 && g.at("mylar_um").get<double>() > 0,
          "Source gap and Mylar thickness must be positive");
  Require(g.at("source_gap_mm").get<double>() + g.at("crystal_mm").at(2).get<double>() / 2
              + g.at("mylar_um").get<double>() * .001 < csi::WorldHalfSizeMm,
          "Source lies outside world");
  Require(g.at("array_nx").get<int>() > 0 && g.at("array_ny").get<int>() > 0, "Invalid array size");
  Require(g.at("channel_active_mm").get<double>() > 0
              && g.at("channel_active_mm").get<double>() <= g.at("channel_pitch_mm").get<double>(),
          "Invalid channel width or pitch");
  Require(g.at("grease_mm").get<double>() > 0 && g.at("ptfe_side_mm").get<double>() > 0
              && g.at("source_radius_mm").get<double>() >= 0,
          "Invalid layer thickness or source radius");
  if (g.contains("source_exit_foil")) {
    const auto& foil = g.at("source_exit_foil");
    const double thickness = foil.at("thickness_mm");
    const double offset = foil.at("distance_from_source_mm");
    const double width = foil.at("width_mm");
    Require(thickness >= 0 && offset > 0 && width > 0 && width < 2*csi::WorldHalfSizeMm,
            "Invalid source-exit foil dimensions");
    Require(offset + thickness < g.at("source_gap_mm").get<double>(),
            "Source-exit foil must lie strictly between source and front Mylar");
    Require(foil.at("material").is_string() && !foil.at("material").get<std::string>().empty(),
            "Source-exit foil requires a NIST material name");
  }
  Require(o.at("charged_step_um").get<double>() > 0
              && o.at("yield_photons_per_MeV").get<double>() >= 0, "Invalid step or light yield");
  for (const std::string key : {"electron_tau_ns", "alpha_tau_ns"}) {
    Require(o.at(key).size() == 2, "Expected two time constants");
    for (double time : o.at(key)) Require(time > 0, "Time constant must be positive");
  }
  for (const std::string key : {"electron_fast_fraction", "alpha_fast_fraction"}) {
    const double fraction = o.at(key);
    Require(fraction >= 0 && fraction <= 1, "Invalid light fraction");
  }
  const auto channels = g.at("summed_channels").get<std::vector<int>>();
  const std::set<int> unique(channels.begin(), channels.end());
  Require(!channels.empty() && unique.size() == channels.size(), "Summed channels must be unique and nonempty");
  for (int channel : channels) {
    Require(channel >= 0 && channel < g.at("array_nx").get<int>() * g.at("array_ny").get<int>(),
            "Invalid channel map");
  }

  // These are internal microcell-index coordinates for the inherited backend,
  // not Geant4 dimensions. Physical hit positions are stored separately.
  candidate["sipm"]["sensor"]["cells_x"] = csi::CellsPerChannel * channels.size();
  candidate["sipm"]["sensor"]["x_max_mm"] = double(csi::CellsPerChannel) * channels.size();
  auto sensor = sipm::Config::FromJSON(candidate.at("sipm").dump());
  Require(sensor.readout.sample_period_ns == a.at("sample_period_ns").get<double>(),
          "ADC sampling definitions disagree");
  const double duration = a.at("samples").get<double>() * a.at("sample_period_ns").get<double>();
  const double pretrigger = a.at("pretrigger_ns");
  Require(pretrigger > 0 && pretrigger < duration, "Decay time must be inside acquisition window");
  Require(sensor.sensor.crosstalk_probability == 0,
          "Spatial crosstalk is disabled: the flattened array map does not preserve neighbors");
  Require(sensor.sensor.cells_y == 1 && sensor.sensor.x_min_mm == 0
              && sensor.sensor.y_min_mm == 0 && sensor.sensor.y_max_mm == 1,
          "Sensor must use the documented flattened cell map");
  sensor.ValidateWindow(duration);

  config_ = std::move(candidate);
  sensor_ = std::move(sensor);
  config_file_ = filename;
}

void RunConfig::SetSource(G4String source) {
  Require(source == "Am241" || source == "Co60", "Only Am241 and Co60 sources are supported");
  source_ = source;
}

void RunConfig::SetSeed(G4long seed) {
  Require(seed > 0 && seed <= 2147483646L, "Seed must be in [1, 2147483646]");
  seed_ = static_cast<unsigned long long>(seed);
  CLHEP::HepRandom::setTheSeed(seed);
}

void RunConfig::SetOutput(G4String filename) {
  Require(!filename.empty(), "Output path must not be empty");
  output_ = filename;
}

void RunConfig::SetExpectedEvents(int events) {
  Require(events > 0, "Events must be positive");
  expected_events_ = events;
}

double RunConfig::DurationNs() const {
  return Acquisition().at("samples").get<double>() * Acquisition().at("sample_period_ns").get<double>();
}

double RunConfig::PretriggerNs() const {
  return Acquisition().at("pretrigger_ns");
}
