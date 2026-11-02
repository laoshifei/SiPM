#include "SpectralVarianceSplit.hh"
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <stdexcept>

namespace {
void Check(bool ok,const char* message) {if (!ok) throw std::runtime_error(message);}
double Mean(const std::vector<double>& x) {return std::accumulate(x.begin(),x.end(),0.)/x.size();}
double Variance(const std::vector<double>& x) {const auto m=Mean(x);double v=0.;for(double y:x)v+=(y-m)*(y-m);return v/x.size();}
}
int main() {
  try {
    std::mt19937_64 rng(62347);std::normal_distribution<double> normal(0.,1.);
    std::vector<double> white(200000);for(double& x:white)x=normal(rng);
    auto original=white;
    nlohmann::json cfg={{"spectral_split",{{"model","complementary_delay_variance"},{"delay_samples",4},{"high_variance_cv",0.}}}};
    SpectralVarianceSplit::Apply(cfg,19881,white);Check(white==original,"Zero split not exact legacy identity");
    for(unsigned delay:{1,4,16}) {
      const auto h=SpectralVarianceSplit::HighInnovations(white,82118,delay);
      std::vector<double> low(h.size());for(std::size_t j=0;j<h.size();++j)low[j]=white[j]-h[j];
      Check(std::abs(Variance(h)-.5)<.012 && std::abs(Variance(low)-.5)<.012,"Incorrect complementary component power");
      for(int lag:{-32,-16,-4,-1,0,1,4,16,32}) {
        double cross=0.;unsigned n=0;
        for(int j=32;j<int(h.size())-32;++j) {cross+=low[j]*h[j+lag];++n;}
        Check(std::abs(cross/n)<.012,"Complementary Gaussian components are cross-correlated");
      }
      double hd=0.;for(std::size_t j=delay;j<h.size();++j)hd+=h[j]*h[j-delay];
      Check(std::abs(hd/(h.size()-delay)+.25)<.012,"High component lag covariance wrong");
    }
    std::vector<double> powers;const unsigned N=256,d=4;const double cv=.6;
    cfg["spectral_split"]["high_variance_cv"]=cv;
    for(unsigned seed=0;seed<6000;++seed) {
      std::vector<double> x(N);for(double& z:x)z=normal(rng);
      auto again=x;SpectralVarianceSplit::Apply(cfg,seed+31300,x);SpectralVarianceSplit::Apply(cfg,seed+31300,again);
      Check(x==again,"Split innovation seed not reproducible");
      double q=0.;for(double z:x)q+=z*z;powers.push_back(q/N);
    }
    const double expected=2./N+cv*cv*(.25+.75/N-.25*d/(double(N)*N));
    Check(std::abs(Mean(powers)-1.)<.025,"Mean spectrum/power changed");
    Check(std::abs(Variance(powers)-expected)<.015,"Split quadratic variance does not match Gaussian mixture moments");
    for(int mode=0;mode<4;++mode) {
      auto bad=cfg;
      if(mode==0)bad["spectral_split"]["delay_samples"]=0;
      if(mode==1)bad["spectral_split"]["delay_samples"]=2.5;
      if(mode==2)bad["spectral_split"]["high_variance_cv"]=-1.;
      if(mode==3)bad["spectral_split"]["model"]="unknown";
      bool rejected=false;try{SpectralVarianceSplit::Validate(bad);}catch(const std::runtime_error&){rejected=true;}
      Check(rejected,"Invalid split settings accepted");
    }
    std::cout<<"PASS exact zero identity, independent complementary components, lag covariance, reproducible seeds, fixed mean power, quadratic mixture variance and invalid parameters\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
