// SPDX-License-Identifier: Apache-2.0
#include "metal_context.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
static void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F> static void rejects(F f){bool rejected=false;try{f();}catch(const std::exception&){rejected=true;}require(rejected,"Invalid prefix clear accepted");}
int main(){try{
 Garnet::MetalContext gpu;size_t checks=0;
 for(size_t rows:{1,2,3})for(size_t groups:{1,3})for(size_t capacity:{1,17,256})for(size_t width:{1,3,128}){
  size_t columns=groups*capacity*width;auto matrix=gpu.Allocate(rows,columns);
  rejects([&]{gpu.ZeroPrefixGroups(matrix,groups,capacity,width,0);});gpu.Zero(matrix);
  for(size_t dirty:{size_t(0),size_t(1),capacity/2,capacity}){
   std::vector<float> values(rows*columns,0);
   for(size_t row=0;row<rows;++row)for(size_t group=0;group<groups;++group)for(size_t i=0;i<dirty*width;++i)
    values[row*columns+group*capacity*width+i]=float(row+group+i+1);
   gpu.Update(matrix,values);auto deviceWritten=gpu.Allocate(rows,columns);
   gpu.Begin();gpu.EncodeAdd(matrix,matrix,deviceWritten);gpu.SubmitAndWait();
   gpu.ZeroPrefixGroups(deviceWritten,groups,capacity,width,dirty);
   for(float value:gpu.Download(deviceWritten)){uint32_t bits;std::memcpy(&bits,&value,4);require(bits==0,"Whole reset buffer must be positive zero");}
   auto consumed=gpu.Allocate(rows,columns);gpu.Begin();gpu.EncodeAdd(deviceWritten,deviceWritten,consumed);gpu.SubmitAndWait();
   for(float value:gpu.Download(consumed))require(value==0,"GPU consumed uncleared state");++checks;
  }
  rejects([&]{gpu.ZeroPrefixGroups(matrix,groups,capacity,width,capacity+1);});
  rejects([&]{gpu.ZeroPrefixGroups(matrix,0,capacity,width,0);});
  rejects([&]{gpu.ZeroPrefixGroups(matrix,groups,capacity,width+1,0);});
  gpu.Begin();rejects([&]{gpu.ZeroPrefixGroups(matrix,groups,capacity,width,0);});gpu.Abort();
 }
 rejects([&]{gpu.ZeroPrefixGroups(Garnet::MetalMatrix(),1,1,1,0);});
 auto source=gpu.Upload(std::vector<float>{1,2,3},1,3);auto aborted=gpu.Allocate(1,3);
 gpu.Begin();gpu.EncodeAdd(source,source,aborted);gpu.Abort();rejects([&]{gpu.ZeroPrefixGroups(aborted,1,3,1,3);});
 gpu.Zero(aborted);gpu.ZeroPrefixGroups(aborted,1,3,1,0);
 std::cout<<"Dirty prefix reset complete CPU/GPU positive zero, bounds, uninitialized and abort PASS "<<checks<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
