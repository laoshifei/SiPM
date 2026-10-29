// Redigitize every saved Geant4 photon history. No transport is regenerated.
// Geometry and optical parameters must match the parent file exactly.
#include "SiPMResponse.hh"
#include "ReadoutNoise.hh"
#include "ReadoutResponse.hh"
#include "Constants.hh"
#include <TFile.h>
#include <TTree.h>
#include <TObjString.h>
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>

int main(int argc, char** argv) {
  try {
    if (argc != 4) throw std::runtime_error("Usage: replay_response input.root candidate.json new.root");
    if (std::filesystem::exists(argv[3])) throw std::runtime_error("Refusing to overwrite replay output");
    TFile input(argv[1], "READ");
    if (input.IsZombie()) throw std::runtime_error("Cannot open parent ROOT file");
    auto* tree = dynamic_cast<TTree*>(input.Get("Events"));
    auto* meta = dynamic_cast<TObjString*>(input.Get("RunInfo"));
    if (!tree || !meta) throw std::runtime_error("Parent lacks Events/RunInfo");
    auto info = nlohmann::json::parse(meta->GetString().Data());
    if (!info.at("completed").get<bool>() || !info.value("digitization_enabled", true)) throw std::runtime_error("Incomplete/transport-only parent");
    auto parent = info.at("config");
    std::ifstream stream(argv[2]); nlohmann::json config; stream >> config;
    if (config.at("geometry") != parent.at("geometry") || config.at("optical") != parent.at("optical")
        || config.at("acquisition") != parent.at("acquisition"))
      throw std::runtime_error("Photon replay requires unchanged geometry, optical and acquisition settings");
    const auto& g = config.at("geometry");
    const auto channels = g.at("summed_channels").get<std::vector<int>>();
    config["sipm"]["sensor"]["cells_x"] = csi::CellsPerChannel*channels.size();
    config["sipm"]["sensor"]["x_max_mm"] = double(csi::CellsPerChannel)*channels.size();
    const auto sensor = sipm::Config::FromJSON(config.at("sipm").dump());
    auto simulated = sensor;
    ReadoutResponse::Validate(config, sensor); ReadoutNoise::Validate(config);
    if (ReadoutNoise::Enabled(config)) simulated.readout.noise_rms_mV = 0.;
    if (sensor.sensor.crosstalk_probability != 0.) throw std::runtime_error("Flattened map does not preserve XT neighbors");
    const double duration = config["acquisition"]["samples"].get<double>()*sensor.readout.sample_period_ns;
    const double pretrigger = config["acquisition"]["pretrigger_ns"];
    const auto seed = info.at("seed").get<std::uint64_t>();
    const int nx = g.at("array_nx"), ny = g.at("array_ny");
    const double pitch = g.at("channel_pitch_mm"), active = g.at("channel_active_mm");
    std::vector<int> group(nx*ny, -1);
    for (std::size_t i=0; i<channels.size(); ++i) group.at(channels[i])=int(i);
    int event_id=0, trigger=0;
    double charge=0., balance=0.;
    std::vector<double> *pt=nullptr,*px=nullptr,*py=nullptr,*pe=nullptr,*at=nullptr,*aq=nullptr,*signal=nullptr,*analog=nullptr,*by_origin=nullptr;
    std::vector<int> *po=nullptr,*pdg=nullptr,*ao=nullptr,*adg=nullptr;
    std::vector<unsigned>* adc=nullptr;
#define BIND(name, ptr) if (tree->SetBranchAddress(name, &ptr)<0) throw std::runtime_error("Missing branch " name)
    BIND("event_id",event_id); BIND("approximate_trigger",trigger); BIND("charge_C",charge); BIND("charge_balance_C",balance);
    BIND("photon_t_ns",pt); BIND("photon_x_mm",px); BIND("photon_y_mm",py); BIND("photon_e_eV",pe);
    BIND("photon_origin",po); BIND("photon_emitter_pdg",pdg);
    BIND("avalanche_t_ns",at); BIND("avalanche_q_C",aq); BIND("avalanche_origin",ao); BIND("avalanche_emitter_pdg",adg);
    BIND("signal_mV",signal); BIND("analog_mV",analog); BIND("adc",adc); BIND("charge_by_origin_C",by_origin);
#undef BIND
    TFile output(argv[3], "CREATE");
    if (output.IsZombie()) throw std::runtime_error("Cannot create replay output");
    auto* target = tree->CloneTree(0);
    info["completed"]=false; info["schema_version"]=8; info["config"]=config;
    info["readout_response"]=config.value("readout_response",nlohmann::json{{"model","legacy_single_path"}});
    info["electronics_noise"]=config.value("electronics_noise",nlohmann::json{{"model","independent_gaussian"}});
    info["response_replay"]={{"parent_file",std::filesystem::absolute(argv[1]).string()},
      {"scope","Original Geant4 optical histories; all entries retained. Native PDE/recovery/AP and readout regenerated, candidate selections must be recomputed."}};
    TObjString(info.dump().c_str()).Write("RunInfo");
    std::uint64_t equal_adc=0, equal_avalanches=0, skipped_zero=0;
    double max_balance=0.;
    const bool same_noise = config.value("electronics_noise",nlohmann::json{}) == parent.value("electronics_noise",nlohmann::json{})
      && config["sipm"]["readout"] == parent["sipm"]["readout"];
    for (Long64_t entry=0; entry<tree->GetEntries(); ++entry) {
      if (tree->GetEntry(entry)<=0) throw std::runtime_error("Cannot read parent event");
      if (pt->empty() && at->empty() && sensor.sensor.dark_rate_hz==0. && same_noise) {
        ++skipped_zero; ++equal_adc; ++equal_avalanches;
      } else {
        std::vector<std::size_t> order(pt->size()); std::iota(order.begin(),order.end(),0);
        std::stable_sort(order.begin(),order.end(),[&](auto a,auto b){return pt->at(a)<pt->at(b);});
        std::vector<sipm::Photon> photons; photons.reserve(order.size());
        for (auto index:order) {
          const double x=px->at(index),y=py->at(index);
          const int ix=int(std::floor(x/pitch+nx/2.)), iy=int(std::floor(y/pitch+ny/2.));
          if (ix<0 || iy<0 || ix>=nx || iy>=ny) throw std::runtime_error("Stored photon outside sensor");
          const double lx=x-(ix-(nx-1)/2.)*pitch, ly=y-(iy-(ny-1)/2.)*pitch;
          const int gr=group.at(iy*nx+ix);
          if (gr<0 || std::abs(lx)>=active/2 || std::abs(ly)>=active/2) throw std::runtime_error("Stored photon outside active channel");
          const int side=csi::EffectiveGridSide;
          const int gx=std::clamp(int((lx/active+.5)*side),0,side-1), gy=std::clamp(int((ly/active+.5)*side),0,side-1);
          const int cell=int((gy*side+gx)*double(csi::CellsPerChannel)/(side*side));
          photons.push_back({pt->at(index),gr*csi::CellsPerChannel+cell+.5,.5,pe->at(index)});
        }
        auto result=sipm::Simulate(simulated,photons,duration,pretrigger,duration,
          sipm::EventSeed(seed+17,event_id),sipm::EventSeed(seed+31,event_id));
        ReadoutResponse::Apply(config,simulated,duration,result);
        ReadoutNoise::Apply(config,sensor.readout,result);
        equal_adc += result.adc_counts == *adc;
        equal_avalanches += result.avalanche_time_ns == *at && result.avalanche_charge_C == *aq;
        ao->clear(); adg->clear(); by_origin->assign(6,0.);
        for (std::size_t k=0;k<result.avalanche_time_ns.size();++k) {
          int origin=0,emitter=0;
          if (result.avalanche_photon_index[k]>=0) {
            auto original=order.at(result.avalanche_photon_index[k]); origin=po->at(original); emitter=pdg->at(original);
          } else if (result.avalanche_parent[k]>=0) {
            origin=ao->at(result.avalanche_parent[k]); emitter=adg->at(result.avalanche_parent[k]);
          }
          ao->push_back(origin); adg->push_back(emitter); by_origin->at(origin)+=result.avalanche_charge_C[k];
        }
        *at=std::move(result.avalanche_time_ns); *aq=std::move(result.avalanche_charge_C);
        *signal=std::move(result.signal_mV); *analog=std::move(result.analog_mV); *adc=std::move(result.adc_counts);
        charge=result.charge_total_C; balance=result.charge_balance_error_C;
        max_balance=std::max(max_balance,std::abs(balance)/std::max(charge,1e-30));
        const auto& r=sensor.readout;
        const double scale=(r.adc_max_mV-r.adc_min_mV)/double(1u<<r.adc_bits);
        trigger=*std::max_element(signal->begin(),signal->end())/scale>=config["acquisition"]["amplitude_threshold_adc"].get<double>();
      }
      if (target->Fill()<0) throw std::runtime_error("Cannot write replay event");
    }
    info["completed"]=target->GetEntries()==tree->GetEntries();
    info["response_replay"]["equal_original_adc_events"]=equal_adc;
    info["response_replay"]["equal_original_avalanche_events"]=equal_avalanches;
    info["response_replay"]["unchanged_zero_photon_events"]=skipped_zero;
    info["response_replay"]["max_relative_charge_balance"]=max_balance;
    target->Write(); TObjString(info.dump().c_str()).Write("RunInfo",TObject::kOverwrite);
    std::cout<<info["response_replay"].dump(2)<<'\n';
  } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
