#include "ReadoutReplica.hh"
#include "ReadoutSampling.hh"
#include "ReadoutNoise.hh"
#include "SignalDependentNoise.hh"
#include <limits>
#include <stdexcept>

ReadoutReplica::ReadoutReplica(const nlohmann::json& job) {
  const auto value=job.value("noise_replica",nlohmann::json(0));
  if (!value.is_number_integer()
      || (!value.is_number_unsigned() && value.get<std::int64_t>()<0)
      || value.get<std::uint64_t>()>std::numeric_limits<std::uint32_t>::max())
    throw std::runtime_error("Noise replica must be an integer in [0,2^32-1]");
  index_=value.get<std::uint64_t>();
  scope_=job.value("noise_scope",std::string("all_readout"));
  if (scope_!="all_readout" && scope_!="signal_only" && scope_!="electronics_only")
    throw std::runtime_error("Unknown noise-replica scope");
}

std::uint64_t ReadoutReplica::Seed(std::uint64_t original) const {
  if (index_==0) return original;
  return sipm::EventSeed(original,0x5245504c00000000ULL+index_);
}

std::uint64_t ReadoutReplica::BaselineSeed(std::uint64_t original) const {
  return scope_=="signal_only" ? original : Seed(original);
}

std::uint64_t ReadoutReplica::SignalSeed(std::uint64_t original) const {
  return scope_=="electronics_only" ? original : Seed(original);
}

void ReadoutReplica::Build(const nlohmann::json& config,const sipm::Config& sensor,
                           double duration,double phase,sipm::Result& result,
                           const sipm::Result* cached_base) const {
  const auto original=result.readout_seed;
  const auto base_seed=BaselineSeed(original);
  auto base_config=config;base_config.erase("signal_dependent_noise");
  if (cached_base) {
    if (cached_base->event_id!=result.event_id || cached_base->readout_seed!=original)
      throw std::runtime_error("Cached base belongs to a different original mother or seed");
    result=*cached_base;
    if (base_seed!=original) {
      result.readout_seed=base_seed;
      if (ReadoutNoise::Enabled(config)) {
        // Rebuild electronics from the unchanged noiseless signal, not by
        // adding another noise realization to the previously noisy ADC.
        ReadoutNoise::Apply(config,sensor.readout,result);
      } else {
        ReadoutSampling::Build(base_config,sensor,duration,phase,result);
      }
    }
  } else {
    result.readout_seed=base_seed;
    ReadoutSampling::Build(base_config,sensor,duration,phase,result);
  }
  result.readout_seed=SignalSeed(original);
  SignalDependentNoise::Apply(config,sensor.readout,result);
}
