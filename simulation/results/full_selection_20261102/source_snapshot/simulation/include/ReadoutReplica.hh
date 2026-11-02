#pragma once
#include "SiPMResponse.hh"
#include <nlohmann/json.hpp>
#include <cstdint>
#include <string>

// Independent downstream noise draws on one unchanged avalanche history.
// Replica zero preserves the original random streams bit for bit.
class ReadoutReplica {
public:
  explicit ReadoutReplica(const nlohmann::json& job);
  std::uint64_t Index() const { return index_; }
  const std::string& Scope() const { return scope_; }
  std::uint64_t BaselineSeed(std::uint64_t original) const;
  std::uint64_t SignalSeed(std::uint64_t original) const;
  void Build(const nlohmann::json&, const sipm::Config&, double duration_ns,
             double phase_ns, sipm::Result&, const sipm::Result* cached_base=nullptr) const;
private:
  std::uint64_t Seed(std::uint64_t original) const;
  std::uint64_t index_=0;
  std::string scope_="all_readout";
};
