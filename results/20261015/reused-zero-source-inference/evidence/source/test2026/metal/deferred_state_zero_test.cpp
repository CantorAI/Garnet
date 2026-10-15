// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F> static void reject(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}need(bad,"Invalid deferred state operation accepted");}
static void positive(const std::vector<float>& v){for(float x:v){uint32_t bits;std::memcpy(&bits,&x,4);need(bits==0,"State is not positive zero");}}
int main(){try{
 Garnet::MetalContext gpu,peer;size_t checks=0;
 for(size_t rows:{1,2,3})for(size_t columns:{1,3,17,65,65536}){
  auto state=gpu.Allocate(rows,columns),out=gpu.Allocate(rows,columns);
  for(float old:{-3.f,std::numeric_limits<float>::quiet_NaN(),std::numeric_limits<float>::infinity()}){
   gpu.Update(state,std::vector<float>(rows*columns,old));gpu.DeferZero(state);gpu.DeferZero(state);
   positive(gpu.Download(state));reject([&]{peer.Download(state);});reject([&]{peer.DeferZero(state);});reject([&]{peer.Update(state,std::vector<float>(rows*columns,7));});
   peer.Begin();reject([&]{peer.EncodeAdd(state,state,out);});peer.Abort();
   gpu.Begin();need(gpu.LastDeferredZeroStates()==1&&gpu.LastDeferredZeroBytes()==rows*columns*4,"Duplicate clear or invalid byte count");
   reject([&]{gpu.Download(state);});reject([&]{gpu.DeferZero(state);});reject([&]{gpu.Update(state,std::vector<float>(rows*columns,9));});
   gpu.EncodeAdd(state,state,out);gpu.SubmitAndWait();positive(gpu.Download(state));positive(gpu.Download(out));positive(peer.Download(state));
   gpu.Begin();need(gpu.LastDeferredZeroStates()==0&&gpu.LastDeferredZeroBytes()==0,"Later inference recleared state");gpu.EncodeAdd(state,state,out);gpu.SubmitAndWait();positive(gpu.Download(out));++checks;
  }
  gpu.DeferZero(state);reject([&]{gpu.Update(state,std::vector<float>(rows*columns+1,1));});positive(gpu.Download(state));
  gpu.Update(state,std::vector<float>(rows*columns,7));gpu.Begin();need(gpu.LastDeferredZeroStates()==0,"Host update did not cancel deferred clear");gpu.EncodeAdd(state,state,out);gpu.SubmitAndWait();need(gpu.Download(out)==std::vector<float>(rows*columns,14),"Canceled reset changed host update");
  gpu.DeferZero(state);gpu.Zero(state);gpu.Begin();need(gpu.LastDeferredZeroStates()==0,"Host zero did not cancel deferred clear");gpu.SubmitAndWait();positive(gpu.Download(state));
  gpu.DeferZero(state);gpu.Abort();reject([&]{gpu.Download(state);});gpu.DeferZero(state);positive(gpu.Download(state));
  gpu.Begin();gpu.EncodeAdd(state,state,out);gpu.Abort();need(gpu.LastDeferredZeroStates()==0&&gpu.LastDeferredZeroBytes()==0,"Abort retained clear statistics");reject([&]{gpu.Download(state);});reject([&]{gpu.Download(out);});
  gpu.DeferZero(state);gpu.Begin();gpu.EncodeAdd(state,state,out);gpu.SubmitAndWait();positive(gpu.Download(out));
  gpu.DeferZero(state);gpu.Begin();gpu.EncodeAdd(state,state,out);gpu.Flush();gpu.Abort();reject([&]{gpu.Download(state);});reject([&]{gpu.Download(out);});
  gpu.DeferZero(state);gpu.Begin();gpu.EncodeAdd(state,state,out);gpu.SubmitAndWait();positive(gpu.Download(out));
 }
 reject([&]{gpu.DeferZero(Garnet::MetalMatrix());});
 auto a=gpu.Upload({1,2,3},1,3),b=gpu.Upload({7,8,9},1,3);gpu.DeferZero(a);gpu.Begin();gpu.SubmitAndWait();positive(gpu.Download(a));need(gpu.Download(b)==std::vector<float>({7,8,9}),"Unrelated state changed");
 Garnet::MetalMatrix abandoned;{Garnet::MetalContext owner;abandoned=owner.Upload({1,2,3},1,3);owner.DeferZero(abandoned);positive(owner.Download(abandoned));}reject([&]{gpu.Download(abandoned);});
 std::cout<<"Deferred reset positive-zero snapshots/GPU use, padded strides, ownership, cancellation, invalid inputs, abort/partial recovery and destruction PASS "<<checks<<" transitions\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
