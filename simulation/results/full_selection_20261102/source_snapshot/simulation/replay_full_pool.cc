// Stream every existing Geant4 mother through frozen, fold-assigned readouts.
// No charge, amplitude, truth, or old-cohort preselection is made here.
#include "ReadoutSampling.hh"
#include "ReadoutReplica.hh"
#include <TFile.h>
#include <TTree.h>
#include <TObjString.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <array>

using json=nlohmann::json;
json read_json(const std::string& path) {
  std::ifstream f(path);if (!f) throw std::runtime_error("Cannot open JSON: "+path);
  json j;f>>j;return j;
}
int main(int argc,char** argv) {
  try {
    if (argc!=3) throw std::runtime_error("Usage: replay_full_pool parent.root plan.json");
    TFile input(argv[1],"READ");
    auto* tree=dynamic_cast<TTree*>(input.Get("Events"));
    auto* object=dynamic_cast<TObjString*>(input.Get("RunInfo"));
    if (input.IsZombie() || !tree || !object) throw std::runtime_error("Invalid parent");
    const auto parent_info=json::parse(object->GetString().Data());
    const auto parent=parent_info.at("config"),plan=read_json(argv[2]);
    const auto seed=parent_info.at("seed").get<std::uint64_t>();
    const auto n=tree->GetEntries();
    const auto folds=plan.at("folds").get<std::vector<int>>();
    if (!parent_info.at("completed").get<bool>() || n!=plan.at("events").get<Long64_t>()
        || folds.size()!=static_cast<std::size_t>(n) || seed!=plan.at("seed").get<std::uint64_t>())
      throw std::runtime_error("Incomplete or mismatched full mother pool");
    std::array<json,2> configs,zero_configs;
    std::array<sipm::Config,2> sensors;
    for (int f=0;f<2;++f) {
      configs[f]=read_json(plan.at("configs").at(f));zero_configs[f]=configs[f];
      zero_configs[f].at("signal_dependent_noise")["event_variance_cv"]=0.;
      for (const auto key:{"geometry","optical","acquisition"})
        if (configs[f].at(key)!=parent.at(key)) throw std::runtime_error("Transport/acquisition changed");
      if (configs[f].at("sipm").at("sensor")!=parent.at("sipm").at("sensor"))
        throw std::runtime_error("Avalanche physics changed");
      sensors[f]=sipm::Config::FromJSON(configs[f].at("sipm").dump());
    }
    const std::string path=plan.at("output");
    if (std::filesystem::exists(path)) throw std::runtime_error("Refusing existing output: "+path);
    int id=0;double charge=0.;std::vector<double>* times=nullptr;std::vector<double>* charges=nullptr;
    std::vector<double>* energy=nullptr;std::vector<unsigned>* old_adc=nullptr;
    tree->SetBranchStatus("*",false);
    for (const auto name:{"event_id","charge_C","avalanche_t_ns","avalanche_q_C","adc","edep_by_origin_MeV"})
      tree->SetBranchStatus(name,true);
    tree->SetBranchAddress("event_id",&id);tree->SetBranchAddress("charge_C",&charge);
    tree->SetBranchAddress("avalanche_t_ns",&times);tree->SetBranchAddress("avalanche_q_C",&charges);
    tree->SetBranchAddress("adc",&old_adc);tree->SetBranchAddress("edep_by_origin_MeV",&energy);
    TFile file(path.c_str(),"CREATE");if (file.IsZombie()) throw std::runtime_error("Cannot create output");
    TTree out("Readout","Full original mother pool; no ADC selection");
    out.SetAutoFlush(-16000000);
    int fold=0;std::vector<unsigned short> original;
    std::vector<double> edep;ULong64_t original_seed=0;
    std::array<std::vector<unsigned short>,4> adc;
    const std::array<std::string,4> names={"current_r0","zero_cv_r0","current_r1","zero_cv_r1"};
    std::array<ULong64_t,4> clipped{},equal_original{};
    out.Branch("event_id",&id);out.Branch("fold",&fold);out.Branch("adc_original",&original);
    out.Branch("edep_by_origin_MeV",&edep);out.Branch("original_readout_seed",&original_seed);
    for (int j=0;j<4;++j) out.Branch(("adc_"+names[j]).c_str(),&adc[j]);
    for (Long64_t i=0;i<n;++i) {
      if (tree->GetEntry(i)<=0 || id!=i) throw std::runtime_error("Missing/nonsequential mother ID");
      fold=folds.at(i);if (fold!=0 && fold!=1) throw std::runtime_error("Invalid fold");
      const auto& c=configs[fold];const auto& sensor=sensors[fold];
      if (sensor.readout.adc_bits>16) throw std::runtime_error("Compact ADC exceeds uint16 range");
      const auto samples=c.at("acquisition").at("samples").get<std::size_t>();
      if (old_adc->size()!=samples) throw std::runtime_error("Parent sample count mismatch");
      const double duration=samples*sensor.readout.sample_period_ns;
      sipm::Result mother;mother.event_id=id;mother.readout_seed=sipm::EventSeed(seed+31,id);
      mother.avalanche_time_ns=*times;mother.avalanche_charge_C=*charges;mother.charge_total_C=charge;
      original_seed=mother.readout_seed;original.assign(old_adc->begin(),old_adc->end());edep=*energy;
      auto base=mother;auto base_config=c;base_config.erase("signal_dependent_noise");
      ReadoutSampling::Build(base_config,sensor,duration,0.,base);
      for (int r=0;r<2;++r) {
        const ReadoutReplica replica(json{{"noise_replica",r},{"noise_scope","all_readout"}});
        for (int v=0;v<2;++v) {
          const int j=2*r+v;
          auto result=mother;
          replica.Build(v==0?c:zero_configs[fold],sensor,duration,0.,result,&base);
          if (result.avalanche_time_ns!=*times || result.avalanche_charge_C!=*charges
              || result.adc_counts.size()!=samples) throw std::runtime_error("Mother history changed");
          clipped[j]+=result.n_clipped_samples;equal_original[j]+=result.adc_counts==*old_adc;
          adc[j].assign(result.adc_counts.begin(),result.adc_counts.end());
        }
      }
      if (out.Fill()<0) throw std::runtime_error("Cannot write full-pool event");
      if ((i+1)%4000==0) std::cout<<plan.at("key")<<" "<<i+1<<"/"<<n<<std::endl;
    }
    json info={{"completed",true},{"schema_version",1},{"parent",std::filesystem::absolute(argv[1]).string()},
      {"seed",seed},{"events",n},{"new_independent_mothers",0},{"plan",plan},{"configs",configs},
      {"zero_configs",zero_configs},{"variants",names},{"clipped_samples",clipped},
      {"equal_original_ADC",equal_original},{"scope","All original mothers, including zero deposits and old QC/charge failures. Frozen fold assignments, replicas 0 and 1. No particle truth or ADC selection. Compact uint16 ADC is lossless; original ADC copied for QC closure."}};
    out.Write();TObjString(info.dump().c_str()).Write("RunInfo");
    std::cout<<plan.at("key")<<" completed "<<n<<" mothers"<<std::endl;
  } catch (const std::exception& e) {std::cerr<<e.what()<<std::endl;return 1;}
}
