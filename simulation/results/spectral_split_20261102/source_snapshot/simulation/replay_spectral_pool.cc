// Complete existing-mother replay for frozen spectral-state candidates.
#include "ReadoutSampling.hh"
#include "ReadoutReplica.hh"
#include <TFile.h>
#include <TTree.h>
#include <TObjString.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <array>

using json=nlohmann::json;
json Read(const std::string& path) {std::ifstream f(path);if(!f)throw std::runtime_error("Cannot read "+path);json j;f>>j;return j;}
int main(int argc,char** argv) {
 try {
  if(argc!=3)throw std::runtime_error("Usage: replay_spectral_pool parent.root plan.json");
  const auto plan=Read(argv[2]);TFile input(argv[1],"READ");
  auto* tree=dynamic_cast<TTree*>(input.Get("Events"));auto* object=dynamic_cast<TObjString*>(input.Get("RunInfo"));
  if(input.IsZombie()||!tree||!object)throw std::runtime_error("Invalid full parent");
  const auto parent_info=json::parse(object->GetString().Data());const auto parent=parent_info.at("config");
  const auto seed=parent_info.at("seed").get<std::uint64_t>();const auto n=tree->GetEntries();
  const auto folds=plan.at("folds").get<std::vector<int>>();
  if(!parent_info.at("completed").get<bool>()||n!=plan.at("events").get<Long64_t>()||folds.size()!=std::size_t(n)||seed!=plan.at("seed").get<std::uint64_t>())
    throw std::runtime_error("Incomplete/mismatched full parent");
  std::vector<std::array<json,2>> configs;std::vector<std::string> names;std::vector<ReadoutReplica> replicas;
  std::set<std::string> unique;
  for(const auto& c:plan.at("cases")) {
    const std::string name=c.at("name");if(!unique.insert(name).second)throw std::runtime_error("Duplicate case");
    names.push_back(name);replicas.emplace_back(c);std::array<json,2> pair;
    for(unsigned f=0;f<2;++f) {
      pair[f]=Read(c.at("configs").at(f));
      for(auto key:{"geometry","optical","acquisition"})if(pair[f].at(key)!=parent.at(key))throw std::runtime_error("Transport/acquisition changed");
      if(pair[f].at("sipm").at("sensor")!=parent.at("sipm").at("sensor"))throw std::runtime_error("Sensor physics changed");
      if(!configs.empty())for(auto key:{"sipm","readout_response","electronics_noise"})
        if(pair[f].at(key)!=configs[0][f].at(key))throw std::runtime_error("Candidate changed mean response or electronics");
    }
    configs.push_back(std::move(pair));
  }
  if(names!=std::vector<std::string>{"current_r0","zero_cv_r0","split_r0","split_r1"})throw std::runtime_error("Unexpected prespecified cases");
  std::array<sipm::Config,2> sensors={sipm::Config::FromJSON(configs[0][0].at("sipm").dump()),sipm::Config::FromJSON(configs[0][1].at("sipm").dump())};
  TFile reference(plan.at("reference").get<std::string>().c_str(),"READ");
  auto* ref=dynamic_cast<TTree*>(reference.Get("Readout"));
  if(reference.IsZombie()||!ref||ref->GetEntries()!=n)throw std::runtime_error("Invalid previous full-pool ADC reference");
  int ref_id=0,ref_fold=0;std::vector<unsigned short>* ref_current=nullptr;std::vector<unsigned short>* ref_zero=nullptr;
  ref->SetBranchStatus("*",false);for(auto name:{"event_id","fold","adc_current_r0","adc_zero_cv_r0"})ref->SetBranchStatus(name,true);
  ref->SetBranchAddress("event_id",&ref_id);ref->SetBranchAddress("fold",&ref_fold);
  ref->SetBranchAddress("adc_current_r0",&ref_current);ref->SetBranchAddress("adc_zero_cv_r0",&ref_zero);
  int id=0,fold=0;double charge=0.;std::vector<double>* times=nullptr;std::vector<double>* charges=nullptr;
  tree->SetBranchStatus("*",false);for(auto name:{"event_id","charge_C","avalanche_t_ns","avalanche_q_C"})tree->SetBranchStatus(name,true);
  tree->SetBranchAddress("event_id",&id);tree->SetBranchAddress("charge_C",&charge);
  tree->SetBranchAddress("avalanche_t_ns",&times);tree->SetBranchAddress("avalanche_q_C",&charges);
  const std::string path=plan.at("output");if(std::filesystem::exists(path))throw std::runtime_error("Refusing existing output");
  TFile file(path.c_str(),"CREATE");if(file.IsZombie())throw std::runtime_error("Cannot create output");
  TTree out("Readout","Full-pool spectral-state candidates; no event preselection");out.SetAutoFlush(-16000000);
  std::vector<std::vector<unsigned short>> adc(names.size());std::vector<ULong64_t> clipped(names.size(),0);
  out.Branch("event_id",&id);out.Branch("fold",&fold);
  for(unsigned j=0;j<names.size();++j)out.Branch(("adc_"+names[j]).c_str(),&adc[j]);
  for(Long64_t i=0;i<n;++i) {
    if(tree->GetEntry(i)<=0||id!=i||ref->GetEntry(i)<=0||ref_id!=id)throw std::runtime_error("Incomplete/nonsequential mother IDs");
    fold=folds.at(i);if((fold!=0&&fold!=1)||ref_fold!=fold)throw std::runtime_error("Fold mismatch");
    const auto& sensor=sensors[fold];const auto samples=configs[0][fold].at("acquisition").at("samples").get<unsigned>();
    if(sensor.readout.adc_bits>16)throw std::runtime_error("Compact ADC cannot represent this bit depth");
    const double duration=samples*sensor.readout.sample_period_ns;
    sipm::Result mother;mother.event_id=id;mother.readout_seed=sipm::EventSeed(seed+31,id);
    mother.avalanche_time_ns=*times;mother.avalanche_charge_C=*charges;mother.charge_total_C=charge;
    auto base=mother;auto c=configs[0][fold];c.erase("signal_dependent_noise");ReadoutSampling::Build(c,sensor,duration,0.,base);
    for(unsigned j=0;j<names.size();++j) {
      auto result=mother;replicas[j].Build(configs[j][fold],sensor,duration,0.,result,&base);
      if(result.avalanche_time_ns!=*times||result.avalanche_charge_C!=*charges||result.adc_counts.size()!=samples)
        throw std::runtime_error("Physical history or sampling changed");
      adc[j].assign(result.adc_counts.begin(),result.adc_counts.end());clipped[j]+=result.n_clipped_samples;
    }
    if(adc[0]!=*ref_current||adc[1]!=*ref_zero)throw std::runtime_error("Legacy full-pool ADC closure failed");
    if(out.Fill()<0)throw std::runtime_error("Cannot write event");
    if((i+1)%8000==0)std::cout<<plan.at("key")<<" "<<i+1<<"/"<<n<<std::endl;
  }
  json info={{"completed",true},{"plan",plan},{"configs",configs},{"events",n},{"seed",seed},
    {"legacy_control_readouts_bit_identical",2*n},{"clipped_samples",clipped},{"new_independent_mothers",0},
    {"scope","Every original Geant4 mother, preserved avalanche histories and mean/electronic response. Original current and zero-CV ADC checked sample-for-sample against previous full-pool output. Candidate selection happens downstream."}};
  out.Write();TObjString(info.dump().c_str()).Write("RunInfo");std::cout<<plan.at("key")<<" complete"<<std::endl;
 }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}
}
