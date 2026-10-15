// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
static void need(bool b,const char* why){if(!b)throw std::runtime_error(why);}
template<class F> static void rejected(F f){bool b=false;try{f();}catch(const std::exception&){b=true;}need(b,"Premature/pending operation accepted");}
int main(){try{Garnet::MetalContext gpu;size_t checks=0;
 rejected([&]{gpu.Flush();});
 for(size_t rows:{1,2,3})for(size_t columns:{1,17,1024})for(size_t chunks:{1,3,7}){
  std::vector<float> values(rows*columns);for(size_t i=0;i<values.size();++i)values[i]=float(int(i%17)-8)/16;
  auto input=gpu.Upload(values,rows,columns);auto baseline=input;
  gpu.Begin();for(size_t i=0;i<chunks+1;++i){auto output=gpu.Allocate(rows,columns);gpu.EncodeAdd(baseline,baseline,output);baseline=output;}gpu.SubmitAndWait();auto expected=gpu.Download(baseline);
  auto current=input;gpu.Begin();
  for(size_t i=0;i<chunks;++i){auto output=gpu.Allocate(rows,columns);gpu.EncodeAdd(current,current,output);current=output;gpu.Flush();rejected([&]{gpu.Download(current);});rejected([&]{gpu.Zero(current);});rejected([&]{gpu.Begin();});}
  auto output=gpu.Allocate(rows,columns);gpu.EncodeAdd(current,current,output);gpu.SubmitAndWait();auto actual=gpu.Download(output);
  need(std::memcmp(actual.data(),expected.data(),actual.size()*4)==0,"Cross-command dependencies changed bits");need(gpu.LastPartialSubmissions()==chunks,"Partial submission count mismatch");++checks;
  gpu.Begin();auto aborted=gpu.Allocate(rows,columns);gpu.EncodeAdd(input,input,aborted);gpu.Flush();auto dropped=gpu.Allocate(rows,columns);gpu.EncodeAdd(aborted,aborted,dropped);gpu.Abort();
  rejected([&]{gpu.Download(aborted);});rejected([&]{gpu.Download(dropped);});gpu.Zero(aborted);gpu.Zero(dropped);
  gpu.Begin();gpu.EncodeAdd(input,input,aborted);gpu.SubmitAndWait();need(gpu.LastPartialSubmissions()==0,"Begin did not reset counts");
  auto recovered=gpu.Download(aborted);for(size_t i=0;i<values.size();++i)need(recovered[i]==values[i]*2,"Abort drain/reuse failed");
 }
 // MPS-to-compute dependency spans two submissions, with a third continuation.
 auto a=gpu.Upload(std::vector<float>{1,2,3,4},2,2);auto b=gpu.Upload(std::vector<float>{1,0,0,1},2,2);auto c=gpu.Allocate(2,2);auto out=gpu.Allocate(2,2);
 gpu.Begin();gpu.EncodeMatMul(a,b,c);gpu.Flush();gpu.EncodeAdd(c,c,out);gpu.Flush();gpu.SubmitAndWait();need(gpu.Download(out)==std::vector<float>({2,4,6,8}),"MPS/compute ordering failed");
 std::cout<<"Prefill stream exact dependencies, ownership, abort drain and recovery PASS "<<checks<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
