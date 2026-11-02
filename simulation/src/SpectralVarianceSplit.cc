#include "SpectralVarianceSplit.hh"
#include "SiPMResponse.hh"
#include <cmath>
#include <random>
#include <stdexcept>

void SpectralVarianceSplit::Validate(const nlohmann::json& noise) {
  if (!noise.contains("spectral_split")) return;
  const auto& split=noise.at("spectral_split");
  if (split.at("model")!="complementary_delay_variance")
    throw std::runtime_error("Unknown spectral variance split");
  const auto& delay=split.at("delay_samples");
  if (!delay.is_number_integer() || delay.get<double>()<1. || delay.get<double>()>64.)
    throw std::runtime_error("Spectral split delay must be an integer in [1,64]");
  const double cv=split.at("high_variance_cv");
  if (!std::isfinite(cv) || cv<0. || cv>2.)
    throw std::runtime_error("Spectral split variance CV must be in [0,2]");
}

std::vector<double> SpectralVarianceSplit::HighInnovations(
    const std::vector<double>& white,std::uint64_t seed,unsigned delay) {
  if (delay<1 || delay>64) throw std::runtime_error("Invalid split delay");
  const auto n=white.size();std::vector<double> w(n+2*delay),v(n+2*delay),high(n);
  std::mt19937_64 edges(sipm::EventSeed(seed,0x53504c49544544ULL));
  std::mt19937_64 auxiliary(sipm::EventSeed(seed,0x53504c49544155ULL));
  std::normal_distribution<double> edge_gaussian(0.,1.),aux_gaussian(0.,1.);
  // Keep every original innovation at its original position; only independent
  // edge extensions are generated. No wraparound or per-event normalization.
  for (unsigned j=0;j<delay;++j) {w[j]=edge_gaussian(edges);w[n+delay+j]=edge_gaussian(edges);}
  for (std::size_t j=0;j<n;++j) w[j+delay]=white[j];
  for (double& x:v) x=aux_gaussian(auxiliary);
  for (std::size_t j=0;j<n;++j)
    high[j]=(-w[j]+2.*w[j+delay]-w[j+2*delay]-v[j]+v[j+2*delay])*.25;
  // In frequency space: P_H=sin^2(omega*d/2), P_L=1-P_H.
  // H=P_H*w-G*v and L=P_L*w+G*v, |G|^2=P_H*P_L.
  // Hence Cov(H,L)=0 at every lag, H+L=w, and PSD(H)=P_H.
  return high;
}

void SpectralVarianceSplit::Apply(const nlohmann::json& noise,
                                  std::uint64_t seed,std::vector<double>& white) {
  if (!noise.contains("spectral_split")) return;
  Validate(noise);const auto& split=noise.at("spectral_split");
  const double cv=split.at("high_variance_cv");
  if (cv==0.) return; // Exact original innovations, including their RNG stream.
  const auto high=HighInnovations(white,seed,split.at("delay_samples").get<unsigned>());
  std::mt19937_64 engine(sipm::EventSeed(seed,0x53504c49545641ULL));
  std::normal_distribution<double> gaussian(0.,1.);
  const double s2=std::log1p(cv*cv);
  const double offset=std::exp(.5*std::sqrt(s2)*gaussian(engine)-.25*s2)-1.;
  for (std::size_t j=0;j<white.size();++j) white[j]+=offset*high[j];
}
