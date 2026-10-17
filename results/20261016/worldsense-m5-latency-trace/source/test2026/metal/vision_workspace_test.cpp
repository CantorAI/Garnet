#include "metal_context.h"
#include "resident_workspace_plan.h"
#include <vector>
#include <cmath>
#include <limits>
#include <iostream>
#include <stdexcept>
using namespace Garnet;
static void need(bool b,const char*m){if(!b)throw std::runtime_error(m);}
static std::vector<float> values(size_t n,float scale){std::vector<float> v(n);for(size_t i=0;i<n;++i)v[i]=float(int(i%17)-8)*scale;return v;}
int main(){try{
 for(unsigned mode=0;mode<16;++mode)need(VisionWorkspaceFullWrite(mode)==(mode==0||mode==2||mode==3||mode==4||mode==5||mode==6||mode==9),"Vision mode admission mismatch");
 size_t count=0;
 for(unsigned mode:{0u,2u,3u,4u,5u,6u,9u})for(size_t batch:{size_t(1),size_t(2)})for(size_t cols:{size_t(7),size_t(12)})for(size_t tokens:{size_t(4),size_t(8)}){
  const size_t hidden=8,heads=2,unit=4,dim=hidden/heads;
  size_t ar=batch*tokens,ac=cols,br=ar,bc=ac,cr=ar,cc=ac,orr=ar,oc=ac;
  if(mode==0){br=1;bc=ac;}
  if(mode==4){orr=ar/unit;oc=ac*unit;}
  if(mode==5){br=4;bc=ac;cr=tokens;cc=8;}
  if(mode==6){ac=3*hidden;oc=ac;br=tokens;bc=dim;cr=ar;cc=ac;}
  if(mode==9){ar=batch;ac=tokens*cols;orr=batch*tokens;oc=cols;br=ar;bc=ac;cr=ar;cc=ac;}
  auto av=values(ar*ac,.03125f),bv=values(br*bc,.015625f),cv=values(cr*cc,.0625f);
  if(mode==5)for(size_t t=0;t<tokens;++t)for(size_t k=0;k<4;++k){cv[t*8+k]=float(k);cv[t*8+4+k]=.25f;}
  if(mode==6)for(size_t i=0;i<bv.size();++i)bv[i]=float(i%4)*.02f;
  MetalContext gpu;auto a=gpu.Upload(av,ar,ac),b=gpu.Upload(bv,br,bc),c=gpu.Upload(cv,cr,cc);
  const size_t n=orr*oc;auto clean=gpu.Allocate(orr,oc),out=gpu.Upload(std::vector<float>(n,std::numeric_limits<float>::quiet_NaN()),orr,oc);
  need(gpu.HazardTracked(out),"Untracked vision scratch");
  gpu.Begin();gpu.EncodeVision(mode,{a,b,c},clean,tokens,hidden,heads,unit);gpu.EncodeVision(mode,{a,b,c},out,tokens,hidden,heads,unit);gpu.SubmitAndWait();
  auto ref=gpu.Download(clean),got=gpu.Download(out);need(got==ref,"Vision read old poisoned output");
  for(size_t row=0;row<orr;++row)for(size_t col=0;col<oc;++col){float x=(mode==4||mode==9)?0.f:av[row*ac+col],want=0;
   if(mode==0)want=x+bv[col];
   if(mode==2)want=.5f*x*(1.f+std::tanh(std::fmax(-10.f,std::fmin(10.f,.7978845608028654f*(x+.044715f*x*x*x)))));
   if(mode==3){float z=std::fabs(x)*.7071067811865475f,t=1.f/(1.f+.3275911f*z);float v=1.f-(((((1.061405429f*t-1.453152027f)*t)+1.421413741f)*t-.284496736f)*t+.254829592f)*t*std::exp(-z*z);want=.5f*x*(1.f+(x<0?-v:v));}
   if(mode==4)want=av[(row*unit+col/ac)*ac+col%ac];
   if(mode==9)want=av[(row/tokens)*ac+(row%tokens)*oc+col];
   if(mode==5){float position=0;for(size_t k=0;k<4;++k)position+=bv[size_t(cv[(row%tokens)*8+k])*bc+col]*cv[(row%tokens)*8+4+k];want=x+position;}
   if(mode==6){if(col>=2*hidden)want=x;else{size_t d=col%dim,base=col-d,other=(d+dim/2)%dim;float angle=bv[(row%tokens)*bc+d];want=x*std::cos(angle)+(d<dim/2?-1.f:1.f)*av[row*ac+base+other]*std::sin(angle);}}
   need(std::isfinite(got[row*oc+col])&&std::fabs(got[row*oc+col]-want)<=3e-6f,"Vision full-output CPU reference mismatch");
  }
  auto av2=av;for(auto&v:av2)v*=2;auto a2=gpu.Upload(av2,ar,ac),clean2=gpu.Allocate(orr,oc),one=gpu.Upload(std::vector<float>(n,1.f),orr,oc),first=gpu.Allocate(orr,oc),second=gpu.Allocate(orr,oc);
  gpu.Begin();gpu.EncodeVision(mode,{a2,b,c},clean2,tokens,hidden,heads,unit);gpu.SubmitAndWait();auto ref2=gpu.Download(clean2);
  gpu.Update(out,std::vector<float>(n,std::numeric_limits<float>::quiet_NaN()));
  gpu.Begin();gpu.EncodeVision(mode,{a,b,c},out,tokens,hidden,heads,unit);gpu.EncodeAdd(out,one,first);gpu.EncodeVision(mode,{a2,b,c},out,tokens,hidden,heads,unit);gpu.EncodeAdd(out,one,second);gpu.SubmitAndWait();auto v1=gpu.Download(first),v2=gpu.Download(second);for(size_t i=0;i<n;++i)need(v1[i]==ref[i]+1.f&&v2[i]==ref2[i]+1.f,"Queued vision reuse overwrote live input or left stale values");
  bool rejected=false;gpu.Begin();try{gpu.EncodeVision(11,{a,b,c},out,tokens,hidden,heads,unit);}catch(const std::invalid_argument&){rejected=true;}gpu.Abort();need(rejected,"Unsupported mode admitted");
  ++count;
 }
 std::cout<<"Vision scalar full-write poison/CPU/queued reuse/tails/merge/batch/invalid checks PASS "<<count<<" cases\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
