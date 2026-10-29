#pragma once

#include "SiPMResponse.hh"
#include "globals.hh"
#include <nlohmann/json.hpp>
#include <memory>
#include <string>

class G4GenericMessenger;

// Shared read-only by transport/actions after /run/initialize. Messenger
// commands are restricted to PreInit; both UI and legacy CLI use these setters.
class RunConfig {
public:
  explicit RunConfig(const std::string& default_config);
  ~RunConfig();

  void Load(G4String filename);
  void SetSource(G4String source);
  void SetSeed(G4long seed);
  void SetOutput(G4String filename);
  void SetExpectedEvents(int events);

  const nlohmann::json& Data() const { return config_; }
  const nlohmann::json& Geometry() const { return config_.at("geometry"); }
  const nlohmann::json& Optical() const { return config_.at("optical"); }
  const nlohmann::json& Acquisition() const { return config_.at("acquisition"); }
  const sipm::Config& SensorModel() const { return sensor_; }
  const std::string& Source() const { return source_; }
  const std::string& Output() const { return output_; }
  const std::string& ConfigFile() const { return config_file_; }
  unsigned long long Seed() const { return seed_; }
  int ExpectedEvents() const { return expected_events_; }
  double DurationNs() const;
  double PretriggerNs() const;

private:
  nlohmann::json config_;
  sipm::Config sensor_{};
  std::string source_ = "Am241";
  std::string output_ = "events.root";
  std::string config_file_;
  unsigned long long seed_ = 1;
  int expected_events_ = 0;
  std::unique_ptr<G4GenericMessenger> messenger_;
};
