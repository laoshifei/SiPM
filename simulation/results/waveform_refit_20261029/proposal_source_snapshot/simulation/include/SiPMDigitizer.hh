#pragma once

#include "SiPMResponse.hh"
#include "G4VDigitizerModule.hh"
#include "G4VDigi.hh"
#include "G4TDigiCollection.hh"

class RunConfig;
struct WaveformEventData;

class SiPMDigi final : public G4VDigi {
public:
  sipm::Result response;
  std::vector<int> avalanche_origin;
  std::vector<int> avalanche_emitter_pdg;
};

using SiPMDigiCollection = G4TDigiCollection<SiPMDigi>;

class SiPMDigitizer final : public G4VDigitizerModule {
public:
  static constexpr const char* Name = "SiPMDigitizer";
  static constexpr const char* Collection = "SiPMDigitizer/SiPMWaveforms";
  explicit SiPMDigitizer(const RunConfig& config);
  void SetInput(const WaveformEventData& event);
  void Digitize() override;

private:
  const RunConfig& config_;
  const WaveformEventData* input_ = nullptr;
};
