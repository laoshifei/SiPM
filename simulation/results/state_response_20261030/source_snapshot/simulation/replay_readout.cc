// Selected-cohort readout diagnostic, reusing physical avalanche histories.
#include "ReadoutSampling.hh"
#include <TFile.h>
#include <TTree.h>
#include <TObjString.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <set>
#include <stdexcept>

using json=nlohmann::json;
json read(const std::string& path) {std::ifstream f(path);json j;f>>j;return j;}
int main(int argc,char** argv) {
  try {
    if(argc!=3)throw std::runtime_error("Usage: replay_readout parent.root jobs.json");
    TFile input(argv[1],"READ");
    auto* tree=dynamic_cast<TTree*>(input.Get("Events"));
    auto* meta=dynamic_cast<TObjString*>(input.Get("RunInfo"));
    if(!tree || !meta || input.IsZombie())throw std::runtime_error("Invalid parent");
    auto info=json::parse(meta->GetString().Data());
    if(!info.at("completed").get<bool>())throw std::runtime_error("Incomplete parent");
    auto plan=read(argv[2]);auto parent=info.at("config");
    const auto wanted=plan.at("ids").get<std::set<int>>();
    const auto seed=info.at("seed").get<std::uint64_t>();
    int id=0;double charge=0.;
    std::vector<double>* times=nullptr;std::vector<double>* charges=nullptr;
    std::vector<unsigned>* old_adc=nullptr;
    tree->SetBranchStatus("*",false);
    for(auto name:{"event_id","charge_C","avalanche_t_ns","avalanche_q_C","adc"})tree->SetBranchStatus(name,true);
    tree->SetBranchAddress("event_id",&id);tree->SetBranchAddress("charge_C",&charge);
    tree->SetBranchAddress("avalanche_t_ns",&times);tree->SetBranchAddress("avalanche_q_C",&charges);
    tree->SetBranchAddress("adc",&old_adc);
    struct Parent {int id;Long64_t entry;sipm::Result result;std::vector<unsigned> adc;};
    std::vector<Parent> cohort;std::set<int> seen;
    for(Long64_t i=0;i<tree->GetEntries();++i) {
      tree->GetBranch("event_id")->GetEntry(i);
      if(!wanted.count(id))continue;
      if(!seen.insert(id).second)throw std::runtime_error("Duplicate parent ID");
      if(tree->GetEntry(i)<=0)throw std::runtime_error("Cannot read selected parent");
      sipm::Result r;r.event_id=id;r.readout_seed=sipm::EventSeed(seed+31,id);
      r.avalanche_time_ns=*times;r.avalanche_charge_C=*charges;r.charge_total_C=charge;
      cohort.push_back({id,i,std::move(r),*old_adc});
    }
    if(seen!=wanted)throw std::runtime_error("Some selected IDs missing");
    for(const auto& job:plan.at("jobs")) {
      const std::string path=job.at("output"),cp=job.at("config");
      if(std::filesystem::exists(path))throw std::runtime_error("Refusing existing output: "+path);
      const auto c=read(cp);
      for(auto key:{"geometry","optical","acquisition"})
        if(c.at(key)!=parent.at(key))throw std::runtime_error("Fixed-history diagnostic changed transport/acquisition");
      if(c.at("sipm").at("sensor")!=parent.at("sipm").at("sensor"))
        throw std::runtime_error("Fixed-avalanche diagnostic changed sensor physics");
      const auto sensor=sipm::Config::FromJSON(c.at("sipm").dump());
      const double duration=c.at("acquisition").at("samples").get<double>()*sensor.readout.sample_period_ns;
      const double phase=job.value("phase_ns",0.);
      TFile file(path.c_str(),"CREATE");if(file.IsZombie())throw std::runtime_error("Cannot create output");
      TTree out("Readout","Selected original-mother readout diagnostic");
      int event_id=0;Long64_t parent_entry=0;std::vector<unsigned> adc;
      std::vector<double> signal,analog;
      out.Branch("event_id",&event_id);out.Branch("parent_entry",&parent_entry);
      out.Branch("adc",&adc);out.Branch("signal_mV",&signal);out.Branch("analog_mV",&analog);
      int equal=0,clipped=0;
      for(const auto& item:cohort) {
        auto r=item.result;ReadoutSampling::Build(c,sensor,duration,phase,r);
        if(r.avalanche_time_ns!=item.result.avalanche_time_ns || r.avalanche_charge_C!=item.result.avalanche_charge_C)
          throw std::runtime_error("Readout changed physical avalanche history");
        if(r.adc_counts.size()!=c.at("acquisition").at("samples").get<std::size_t>())throw std::runtime_error("Sample count changed");
        equal+=r.adc_counts==item.adc;clipped+=r.n_clipped_samples;
        event_id=item.id;parent_entry=item.entry;adc=std::move(r.adc_counts);
        signal=std::move(r.signal_mV);analog=std::move(r.analog_mV);
        if(out.Fill()<0)throw std::runtime_error("Cannot write event");
      }
      json record={{"completed",true},{"config",c},{"phase_ns",phase},{"parent",std::filesystem::absolute(argv[1]).string()},
          {"n_selected_original_mothers",cohort.size()},{"new_independent_mothers",0},{"equal_parent_ADC",equal},
          {"clipped_samples",clipped},{"seed",seed},{"ids",wanted},
          {"scope","Fixed original avalanche histories, independent downstream streams. Phase diagnostics extend noiseless integration by one sample, without new avalanches; record-end region is censored and excluded from comparisons."}};
      if(job.value("require_identity",false) && equal!=int(cohort.size()))throw std::runtime_error("Identity ADC closure failed");
      out.Write();TObjString(record.dump().c_str()).Write("RunInfo");
      std::cout<<job.at("key")<<" "<<cohort.size()<<" original mothers, identity ADC="<<equal<<'\n'<<std::flush;
    }
  } catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
