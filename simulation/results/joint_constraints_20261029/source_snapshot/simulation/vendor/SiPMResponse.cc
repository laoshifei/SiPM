#include "SiPMResponse.hh"

#include <nlohmann/json.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>

namespace sipm {
namespace {
void Require(bool ok, const std::string& message) {
  if(!ok) throw std::runtime_error("SiPM: " + message);
}
bool Nonnegative(double x) { return std::isfinite(x) && x >= 0.; }
bool Positive(double x) { return std::isfinite(x) && x > 0.; }
bool Probability(double x) { return Nonnegative(x) && x <= 1.; }
std::uint64_t Mix(std::uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  return x ^ (x >> 31);
}
std::size_t IntegerRatio(double a, double b, const char* name) {
  const double n = a / b;
  Require(Positive(n) && n <= static_cast<double>(std::numeric_limits<int>::max()) &&
          std::abs(n - std::round(n)) < 1e-8, std::string(name) + " must be an integer multiple of internal_dt_ns");
  return static_cast<std::size_t>(std::llround(n));
}
void Axis(const std::vector<double>& axis, const char* name) {
  Require(axis.size() >= 2, std::string(name) + " needs at least two entries");
  for(std::size_t i = 0; i < axis.size(); ++i)
    Require(Nonnegative(axis[i]) && (i == 0 || axis[i] > axis[i-1]), std::string(name) + " must increase strictly");
}
std::size_t Interval(const std::vector<double>& axis, double x) {
  Require(std::isfinite(x) && x >= axis.front() && x <= axis.back(), "PDE extrapolation is forbidden");
  return std::min(static_cast<std::size_t>(std::upper_bound(axis.begin(), axis.end(), x) - axis.begin() - 1), axis.size()-2);
}
double Uniform(std::mt19937_64& engine) { return std::generate_canonical<double, 53>(engine); }
double Exponential(std::mt19937_64& engine, double mean) {
  return -mean * std::log1p(-Uniform(engine));
}
}

Config Config::FromJSON(const std::string& text) {
  const auto j = nlohmann::json::parse(text);
  Require(j.at("schema_version") == 1, "unsupported config schema_version");
  Config c;
  c.json_text = text;
  const auto& s = j.at("sensor");
  const auto& r = j.at("readout");
#define SENSOR(field) s.at(#field).get_to(c.sensor.field)
  SENSOR(name); SENSOR(parameter_status); SENSOR(parameter_source); SENSOR(temperature_C);
  SENSOR(cells_x); SENSOR(cells_y); SENSOR(x_min_mm); SENSOR(x_max_mm); SENSOR(y_min_mm); SENSOR(y_max_mm);
  SENSOR(overvoltage_V); SENSOR(gain_electrons); SENSOR(recovery_ns); SENSOR(trigger_scale_V);
  SENSOR(trigger_threshold_V); SENSOR(recovery_enabled); SENSOR(pde_mode);
  SENSOR(pde_wavelength_nm); SENSOR(pde_overvoltage_V); SENSOR(pde_probabilities);
  if(!s.at("constant_pde").is_null()) SENSOR(constant_pde);
  SENSOR(dark_rate_hz); SENSOR(crosstalk_probability); SENSOR(crosstalk_delay_ns);
  SENSOR(afterpulse_probability); SENSOR(afterpulse_delay_ns);
  SENSOR(transit_delay_mean_ns); SENSOR(transit_delay_sigma_ns); SENSOR(max_candidates);
#undef SENSOR
  Require(r.at("model") == "double_exponential", "native v1 supports only double_exponential readout");
  Require(r.at("template_csv").is_null(), "native v1 does not support measured templates");
#define READOUT(field) r.at(#field).get_to(c.readout.field)
  READOUT(internal_dt_ns); READOUT(sample_period_ns); READOUT(aperture_ns);
  READOUT(rise_ns); READOUT(decay_ns); READOUT(transimpedance_ohm); READOUT(lowpass_ns);
  READOUT(baseline_mV); READOUT(noise_rms_mV); READOUT(adc_bits); READOUT(adc_min_mV);
  READOUT(adc_max_mV); READOUT(max_internal_bins);
#undef READOUT
  const auto& a = c.sensor;
  const auto& b = c.readout;
  Require(!a.name.empty() && !a.parameter_source.empty(), "name and parameter_source are required");
  Require(a.parameter_status == "hypothetical" || a.parameter_status == "datasheet_based" ||
          a.parameter_status == "calibrated", "invalid parameter_status");
  Require(std::isfinite(a.temperature_C), "invalid temperature_C");
  Require(a.cells_x > 0 && a.cells_y > 0 && static_cast<std::int64_t>(a.cells_x)*a.cells_y <= 10000000,
          "cell count must be in [1,10000000]");
  Require(std::isfinite(a.x_min_mm) && std::isfinite(a.x_max_mm) && a.x_max_mm > a.x_min_mm &&
          std::isfinite(a.y_min_mm) && std::isfinite(a.y_max_mm) && a.y_max_mm > a.y_min_mm, "invalid sensor bounds");
  Require(Positive(a.overvoltage_V) && Positive(a.gain_electrons) && Positive(a.recovery_ns) &&
          Positive(a.trigger_scale_V) && Nonnegative(a.trigger_threshold_V) && a.trigger_threshold_V < a.overvoltage_V,
          "invalid gain, recovery or trigger parameters");
  Require(a.pde_mode == "table" || a.pde_mode == "constant", "invalid pde_mode");
  if(a.pde_mode == "constant") {
    Require(!s.at("constant_pde").is_null() && Probability(a.constant_pde), "constant_pde must be explicit and in [0,1]");
  } else {
    Axis(a.pde_wavelength_nm, "PDE wavelengths"); Axis(a.pde_overvoltage_V, "PDE overvoltages");
    Require(a.pde_wavelength_nm.front() > 0. && a.pde_overvoltage_V.front() == 0. &&
            a.pde_overvoltage_V.back() >= a.overvoltage_V, "PDE axes must cover wavelength and [0,overvoltage]");
    Require(a.pde_probabilities.size() == a.pde_overvoltage_V.size(), "PDE row count mismatch");
    for(std::size_t v=0; v<a.pde_probabilities.size(); ++v) {
      Require(a.pde_probabilities[v].size() == a.pde_wavelength_nm.size(), "PDE column count mismatch");
      for(std::size_t w=0; w<a.pde_wavelength_nm.size(); ++w) {
        const double p = a.pde_probabilities[v][w];
        Require(Probability(p) && (v ? p >= a.pde_probabilities[v-1][w] : p == 0.),
                "PDE must start at zero and be nondecreasing with overvoltage");
      }
    }
  }
  Require(Nonnegative(a.dark_rate_hz) && Probability(a.crosstalk_probability) && Probability(a.afterpulse_probability) &&
          a.crosstalk_probability+a.afterpulse_probability < 1., "invalid noise rates or supercritical branching");
  Require(Nonnegative(a.crosstalk_delay_ns) && Positive(a.afterpulse_delay_ns) &&
          Nonnegative(a.transit_delay_mean_ns) && Nonnegative(a.transit_delay_sigma_ns) &&
          (a.transit_delay_sigma_ns == 0. || a.transit_delay_mean_ns > 0.), "invalid timing parameters");
  Require(s.at("max_candidates").is_number_integer() && s.at("max_candidates").get<double>() > 0. &&
          a.max_candidates <= static_cast<std::uint64_t>(std::numeric_limits<int>::max()), "invalid max_candidates");
  Require(Positive(b.internal_dt_ns) && Positive(b.sample_period_ns) && Positive(b.aperture_ns) &&
          b.aperture_ns <= b.sample_period_ns, "invalid sampling settings");
  IntegerRatio(b.sample_period_ns,b.internal_dt_ns,"sample_period_ns");
  IntegerRatio(b.aperture_ns,b.internal_dt_ns,"aperture_ns");
  Require(Positive(b.rise_ns) && Positive(b.decay_ns) && b.decay_ns > b.rise_ns && Positive(b.transimpedance_ohm) &&
          Nonnegative(b.lowpass_ns) && std::isfinite(b.baseline_mV) && Nonnegative(b.noise_rms_mV), "invalid readout parameters");
  Require(b.adc_bits >= 1 && b.adc_bits <= 24 && std::isfinite(b.adc_min_mV) && std::isfinite(b.adc_max_mV) &&
          b.adc_max_mV > b.adc_min_mV, "invalid ADC range or bits");
  Require(r.at("max_internal_bins").is_number_integer() && r.at("max_internal_bins").get<double>() > 0. &&
          b.max_internal_bins <= 100000000, "invalid max_internal_bins");
  return c;
}
void Config::ValidateWindow(double window_ns) const {
  const auto n = IntegerRatio(window_ns,readout.internal_dt_ns,"window_ns");
  const auto stride = IntegerRatio(readout.sample_period_ns,readout.internal_dt_ns,"sample_period_ns");
  Require(n <= readout.max_internal_bins && n % stride == 0, "window must fit memory limit and whole sample periods");
}
std::uint64_t EventSeed(std::uint64_t base, std::uint64_t event_id) { return Mix(base ^ Mix(event_id)); }
int MapCell(const Sensor& s, double x, double y) {
  if(!std::isfinite(x) || !std::isfinite(y) || x<s.x_min_mm || x>s.x_max_mm || y<s.y_min_mm || y>s.y_max_mm) return -1;
  const int ix = std::min(s.cells_x-1, static_cast<int>((x-s.x_min_mm)/(s.x_max_mm-s.x_min_mm)*s.cells_x));
  const int iy = std::min(s.cells_y-1, static_cast<int>((y-s.y_min_mm)/(s.y_max_mm-s.y_min_mm)*s.cells_y));
  return iy*s.cells_x+ix;
}
double Recovery(const Sensor& s, double dt) {
  return s.recovery_enabled ? -std::expm1(-std::max(0.,dt)/s.recovery_ns) : 1.;
}
double TriggerFraction(const Sensor& s, double r) {
  return -std::expm1(-std::max(0.,r*s.overvoltage_V-s.trigger_threshold_V)/s.trigger_scale_V) /
         -std::expm1(-(s.overvoltage_V-s.trigger_threshold_V)/s.trigger_scale_V);
}
double PhotonPDE(const Sensor& s, double wavelength, double r) {
  if(s.pde_mode == "constant") return s.constant_pde*TriggerFraction(s,r);
  const auto i = Interval(s.pde_wavelength_nm,wavelength);
  const double voltage = r*s.overvoltage_V;
  const auto j = Interval(s.pde_overvoltage_V,voltage);
  const double w = (wavelength-s.pde_wavelength_nm[i])/(s.pde_wavelength_nm[i+1]-s.pde_wavelength_nm[i]);
  const double v = (voltage-s.pde_overvoltage_V[j])/(s.pde_overvoltage_V[j+1]-s.pde_overvoltage_V[j]);
  const auto& p = s.pde_probabilities;
  return (1-v)*((1-w)*p[j][i]+w*p[j][i+1]) + v*((1-w)*p[j+1][i]+w*p[j+1][i+1]);
}

Result Simulate(const Config& cfg, const std::vector<Photon>& photons, double window,
                double gate_start, double gate_end, std::uint64_t seed, std::uint64_t read_seed) {
  cfg.ValidateWindow(window);
  Require(Nonnegative(gate_start) && std::isfinite(gate_end) && gate_start < gate_end && gate_end <= window,"invalid analysis gate");
  Require(photons.size() <= static_cast<std::size_t>(std::numeric_limits<int>::max()),"too many input photons");
  const auto& s = cfg.sensor;
  Result out;
  out.avalanche_seed=seed; out.readout_seed=read_seed; out.n_input_photons=photons.size();
  std::mt19937_64 photon_rng(EventSeed(seed,0)), transit_rng(EventSeed(seed,1)), dark_rng(EventSeed(seed,2));
  std::mt19937_64 secondary_rng(EventSeed(seed,3)), noise_rng(EventSeed(seed,4));
  struct Candidate { double time; std::uint64_t order; int cell,kind,parent,index; };
  struct Later { bool operator()(const Candidate& a,const Candidate& b) const {
    return a.time != b.time ? a.time > b.time : a.order > b.order;
  }};
  std::priority_queue<Candidate,std::vector<Candidate>,Later> queue;
  std::vector<Candidate> incoming;
  incoming.reserve(photons.size());
  std::uint64_t order=0;
  double previous=-1.;
  const double gamma_shape = s.transit_delay_sigma_ns > 0. ? std::pow(s.transit_delay_mean_ns/s.transit_delay_sigma_ns,2) : 1.;
  const double gamma_scale = s.transit_delay_sigma_ns > 0. ? std::pow(s.transit_delay_sigma_ns,2)/s.transit_delay_mean_ns : 1.;
  std::gamma_distribution<double> transit(gamma_shape,gamma_scale);
  for(std::size_t i=0; i<photons.size(); ++i) {
    const auto& p=photons[i];
    Require(Nonnegative(p.time_ns) && p.time_ns >= previous && p.time_ns < window && Positive(p.energy_eV) &&
            std::isfinite(p.x_mm) && std::isfinite(p.y_mm), "input photons must be finite, sorted and inside [0,T)");
    previous=p.time_ns;
    const int cell=MapCell(s,p.x_mm,p.y_mm);
    if(cell<0) { ++out.n_outside_sensor; continue; }
    PhotonPDE(s,hc_eV_nm/p.energy_eV,1.); // Check coverage even if the delayed photon falls outside T.
    const double time=p.time_ns+(s.transit_delay_sigma_ns>0. ? transit(transit_rng) : s.transit_delay_mean_ns);
    if(time>=window) { ++out.n_delayed_photons_outside; continue; }
    incoming.push_back({time,order++,cell,photon,-1,static_cast<int>(i)});
  }
  std::stable_sort(incoming.begin(),incoming.end(),[](const auto& a,const auto& b){ return a.time<b.time; });
  const int cells=s.cells_x*s.cells_y;
  std::uniform_int_distribution<int> random_cell(0,cells-1);
  auto next_dark = [&](double start) {
    if(s.dark_rate_hz==0.) return;
    const double t=start+Exponential(dark_rng,1e9/s.dark_rate_hz);
    if(t<window) queue.push({t,order++,random_cell(dark_rng),dark,-1,-1});
  };
  auto secondary = [&](double time,int cell,int kind,int parent) {
    if(time>=window) { ++out.n_secondary_outside; return; }
    queue.push({time,order++,cell,kind,parent,-1});
  };
  next_dark(0.);
  std::vector<double> last_fire(cells,-std::numeric_limits<double>::infinity());
  std::size_t i=0;
  while(i<incoming.size() || !queue.empty()) {
    Candidate c;
    if(!queue.empty() && (i==incoming.size() || queue.top().time<=incoming[i].time)) { c=queue.top(); queue.pop(); }
    else c=incoming[i++];
    Require(++out.n_candidates<=s.max_candidates,"max_candidates exceeded; incomplete response is not saved");
    if(c.kind==dark) next_dark(c.time);
    const double recovery=Recovery(s,c.time-last_fire[c.cell]);
    const double probability=c.kind==photon ? PhotonPDE(s,hc_eV_nm/photons[c.index].energy_eV,recovery) : TriggerFraction(s,recovery);
    if(Uniform(c.kind==photon ? photon_rng : noise_rng)>=probability) continue;
    const int parent=static_cast<int>(out.avalanche_time_ns.size());
    const double q=electron_charge_C*s.gain_electrons*recovery;
    last_fire[c.cell]=c.time;
    out.avalanche_time_ns.push_back(c.time); out.avalanche_charge_C.push_back(q);
    out.avalanche_recovery.push_back(recovery); out.avalanche_cell.push_back(c.cell);
    out.avalanche_kind.push_back(c.kind); out.avalanche_parent.push_back(c.parent); out.avalanche_photon_index.push_back(c.index);
    out.charge_total_C+=q;
    if(c.time>=gate_start && c.time<gate_end) out.charge_gate_C+=q;
    if(c.kind==photon) ++out.n_photon_avalanches;
    if(c.kind==dark) ++out.n_dark_avalanches;
    if(c.kind==crosstalk) ++out.n_crosstalk_avalanches;
    if(c.kind==afterpulse) ++out.n_afterpulse_avalanches;
    // First-stage approximation: at most one XT and one AP proposal per avalanche.
    // Every accepted secondary may itself branch. No particle label enters this model.
    if(Uniform(secondary_rng)<s.crosstalk_probability*recovery) {
      std::array<int,4> neighbors{}; int count=0;
      const int x=c.cell%s.cells_x,y=c.cell/s.cells_x;
      if(x>0) neighbors[count++]=c.cell-1;
      if(x+1<s.cells_x) neighbors[count++]=c.cell+1;
      if(y>0) neighbors[count++]=c.cell-s.cells_x;
      if(y+1<s.cells_y) neighbors[count++]=c.cell+s.cells_x;
      if(count) {
        const int cell=neighbors[std::uniform_int_distribution<int>(0,count-1)(secondary_rng)];
        const double delay=s.crosstalk_delay_ns>0. ? Exponential(secondary_rng,s.crosstalk_delay_ns) : 0.;
        secondary(c.time+delay,cell,crosstalk,parent);
      }
    }
    if(Uniform(secondary_rng)<s.afterpulse_probability*recovery)
      secondary(c.time+Exponential(secondary_rng,s.afterpulse_delay_ns),c.cell,afterpulse,parent);
  }
  BuildReadout(cfg,window,out);
  return out;
}

void BuildReadout(const Config& cfg, double window, Result& out) {
  cfg.ValidateWindow(window);
  const auto& r=cfg.readout;
  const auto n=IntegerRatio(window,r.internal_dt_ns,"window_ns");
  const auto stride=IntegerRatio(r.sample_period_ns,r.internal_dt_ns,"sample_period_ns");
  const auto aperture=IntegerRatio(r.aperture_ns,r.internal_dt_ns,"aperture_ns");
  Require(out.avalanche_time_ns.size()==out.avalanche_charge_C.size(),"avalanche array length mismatch");
  double last=-1.;
  for(std::size_t k=0;k<out.avalanche_time_ns.size();++k) {
    const double t=out.avalanche_time_ns[k];
    Require(Nonnegative(t) && t>=last && t<window && Nonnegative(out.avalanche_charge_C[k]),"invalid avalanche input to readout");
    last=t;
  }
  out.current_time_ns.resize(n); out.current_A.resize(n);
  out.sample_time_ns.clear(); out.signal_mV.clear(); out.analog_mV.clear(); out.adc_counts.clear(); out.adc_clipped.clear();
  out.current_integral_C=0.; out.current_tail_C=0.; out.n_clipped_samples=0;
  const double dt=r.internal_dt_ns;
  const double ar=std::exp(-dt/r.rise_ns), ad=std::exp(-dt/r.decay_ns);
  const double br=-std::expm1(-dt/r.rise_ns), bd=-std::expm1(-dt/r.decay_ns);
  const double al=r.lowpass_ns>0. ? std::exp(-dt/r.lowpass_ns) : 0.;
  const double fraction=r.lowpass_ns>0. ? r.lowpass_ns*(-std::expm1(-dt/r.lowpass_ns))/dt : 0.;
  const unsigned levels=1u<<r.adc_bits;
  const double lsb=(r.adc_max_mV-r.adc_min_mV)/levels;
  std::mt19937_64 engine(out.readout_seed);
  std::normal_distribution<double> gaussian(0.,1.);
  double sr=0.,sd=0.,lowpass_state=0.,sample_sum=0.;
  std::size_t k=0;
  for(std::size_t b=0;b<n;++b) {
    const double end=(b+1)*dt;
    double ir=0.,id=0.,er=0.,ed=0.;
    while(k<out.avalanche_time_ns.size() && out.avalanche_time_ns[k]<end) {
      const double left=end-out.avalanche_time_ns[k],q=out.avalanche_charge_C[k];
      ir+=q*(-std::expm1(-left/r.rise_ns)); id+=q*(-std::expm1(-left/r.decay_ns));
      er+=q*std::exp(-left/r.rise_ns); ed+=q*std::exp(-left/r.decay_ns);
      ++k;
    }
    // Exact bin integral of q*(exp(-t/decay)-exp(-t/rise))/(decay-rise).
    const double bin_charge=(r.decay_ns*(sd*bd+id)-r.rise_ns*(sr*br+ir))/(r.decay_ns-r.rise_ns);
    sr=ar*sr+er; sd=ad*sd+ed;
    out.current_time_ns[b]=(b+0.5)*dt;
    out.current_A[b]=bin_charge/(dt*1e-9);
    out.current_integral_C+=bin_charge;
    const double input_mV=out.current_A[b]*r.transimpedance_ohm*1000.;
    const double voltage=fraction*lowpass_state+(1.-fraction)*input_mV;
    lowpass_state=al*lowpass_state+(1.-al)*input_mV;
    if(b%stride==0) sample_sum=0.;
    if(b%stride<aperture) sample_sum+=voltage;
    if(b%stride==aperture-1) {
      const double signal=sample_sum/aperture;
      const double analog=signal+r.baseline_mV+(r.noise_rms_mV>0. ? r.noise_rms_mV*gaussian(engine) : 0.);
      const bool clipped=analog<r.adc_min_mV || analog>=r.adc_max_mV;
      const double code=std::floor((analog-r.adc_min_mV)/lsb);
      out.sample_time_ns.push_back((b/stride)*r.sample_period_ns+0.5*r.aperture_ns);
      out.signal_mV.push_back(signal); out.analog_mV.push_back(analog);
      out.adc_counts.push_back(static_cast<unsigned>(std::clamp(code,0.,static_cast<double>(levels-1))));
      out.adc_clipped.push_back(clipped); out.n_clipped_samples+=clipped;
    }
  }
  for(std::size_t a=0;a<out.avalanche_time_ns.size();++a) {
    const double left=window-out.avalanche_time_ns[a];
    out.current_tail_C+=out.avalanche_charge_C[a]*(r.decay_ns*std::exp(-left/r.decay_ns)-
      r.rise_ns*std::exp(-left/r.rise_ns))/(r.decay_ns-r.rise_ns);
  }
  out.charge_balance_error_C=out.current_integral_C+out.current_tail_C-out.charge_total_C;
}
}
