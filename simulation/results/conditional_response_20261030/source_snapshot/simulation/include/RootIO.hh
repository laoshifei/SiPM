#pragma once

#include <nlohmann/json.hpp>
#include <memory>

class RunConfig;
class TFile;
class TTree;
struct WaveformEventData;

class RootIO {
public:
  RootIO(const RunConfig& config, std::shared_ptr<WaveformEventData> data);
  ~RootIO();
  void Open(int requested_events);
  void FillEvent();
  void Close(int completed_events);

private:
  void ReleaseFile();
  const RunConfig& config_;
  std::shared_ptr<WaveformEventData> data_;
  std::unique_ptr<TFile> file_;
  std::unique_ptr<TTree> tree_;
  nlohmann::json metadata_;
};
