// Exact EM equations used by shape_mixture.py; no fast-math or changed model.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

extern "C" void publication_em(const double* x, const int* group, int n,
    const double* seed, double center, double scale, double floor, int max_iter,
    double tolerance, double* out) {
  const double pi = std::acos(-1.0), norm = std::sqrt(2*pi);
  double mu[2] = {seed[0], seed[1]}, sd[2] = {std::max(scale/8, floor), std::max(scale/8, floor)};
  // Starting widths use the unscaled broad width, supplied separately in seed[2].
  sd[0] = sd[1] = std::max(seed[2]/8, floor);
  double w[2][3] = {{.475,.475,.05},{.475,.475,.05}};
  std::vector<double> resp(3*n), bg(n);
  int ng[2] = {0,0};
  for (int i=0;i<n;++i) {double z=(x[i]-center)/scale; bg[i]=1/(pi*scale*(1+z*z)); ++ng[group[i]];}
  auto evaluate = [&]() {
    double ll=0;
    for (int i=0;i<n;++i) {
      double total=0;
      for (int k=0;k<2;++k) {double z=(x[i]-mu[k])/sd[k]; resp[3*i+k]=std::exp(-.5*z*z)/(norm*sd[k])*w[group[i]][k]; total+=resp[3*i+k];}
      resp[3*i+2]=bg[i]*w[group[i]][2]; total+=resp[3*i+2]; total=std::max(total,1e-300);
      ll+=std::log(total);
      for (int k=0;k<3;++k) resp[3*i+k]/=total;
    }
    return ll;
  };
  double previous=-std::numeric_limits<double>::infinity(), minstep=std::numeric_limits<double>::infinity();
  bool converged=false; int iteration=0;
  for (;iteration<max_iter;++iteration) {
    double ll=evaluate();
    if (iteration) minstep=std::min(minstep,ll-previous);
    if (iteration && std::abs(ll-previous)/n<tolerance) {converged=true;break;}
    previous=ll;
    double sums[2]={0,0}, means[2]={0,0}, var[2]={0,0}, nw[2][3]={{0,0,0},{0,0,0}};
    for (int i=0;i<n;++i) {
      for (int k=0;k<2;++k) {sums[k]+=resp[3*i+k]; means[k]+=resp[3*i+k]*x[i];}
      for (int k=0;k<3;++k) nw[group[i]][k]+=resp[3*i+k];
    }
    for (int k=0;k<2;++k) means[k]/=std::max(sums[k],1e-100);
    for (int i=0;i<n;++i) for (int k=0;k<2;++k) {double d=x[i]-means[k]; var[k]+=resp[3*i+k]*d*d;}
    int order[3]={0,1,2}; if (means[1]<means[0]) std::swap(order[0],order[1]);
    for (int k=0;k<2;++k) {int j=order[k];mu[k]=means[j];sd[k]=std::max(std::sqrt(var[j]/std::max(sums[j],1e-100)),floor);}
    for (int g=0;g<2;++g) {
      double total=0;
      for (int k=0;k<3;++k) {w[g][k]=std::max(nw[g][order[k]]/ng[g],1e-12);total+=w[g][k];}
      for (int k=0;k<3;++k) w[g][k]/=total;
    }
  }
  out[0]=mu[0];out[1]=mu[1];out[2]=sd[0];out[3]=sd[1];
  for (int g=0;g<2;++g) for (int k=0;k<3;++k) out[4+3*g+k]=w[g][k];
  out[10]=evaluate();out[11]=converged;out[12]=std::min(iteration+1,max_iter);out[13]=minstep;
}
