// Correctness only: native compact-prefix byte copies, ownership and bounds.
#include "metal_context.h"
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
static void need(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
template<class F>static void reject(F f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}need(bad,"Invalid prefix access accepted");}
int main(){try{Garnet::MetalContext gpu;size_t checks=0;
 for(size_t rows:{1,2,3})for(size_t groups:{1,3})for(size_t capacity:{65,128})for(size_t width:{1,3,128}){
  size_t columns=groups*capacity*width;std::vector<float> values(rows*columns),expected;
  for(size_t i=0;i<values.size();++i)values[i]=float(int(i%29)-14);
  values[0]=-0.0f;auto source=gpu.Upload(values,rows,columns),target=gpu.Allocate(rows,columns);
  reject([&]{gpu.ReadPrefixGroups(target,groups,capacity,width,64);});
  reject([&]{gpu.WritePrefixGroups(target,groups,capacity,width,64,std::vector<float>(rows*groups*64*width));});
  for(size_t row=0;row<rows;++row)for(size_t group=0;group<groups;++group)
   expected.insert(expected.end(),values.begin()+row*columns+group*capacity*width,values.begin()+row*columns+group*capacity*width+64*width);
  auto prefix=gpu.ReadPrefixGroups(source,groups,capacity,width,64);need(prefix.size()==expected.size()&&!std::memcmp(prefix.data(),expected.data(),prefix.size()*4),"Native prefix capture bits changed");
  std::vector<float> full(rows*columns,99);gpu.Update(target,full);gpu.WritePrefixGroups(target,groups,capacity,width,64,prefix);
  for(size_t row=0;row<rows;++row)for(size_t group=0;group<groups;++group)
   std::memcpy(full.data()+row*columns+group*capacity*width,prefix.data()+(row*groups+group)*64*width,64*width*4);
  auto actual=gpu.Download(target);need(!std::memcmp(actual.data(),full.data(),full.size()*4),"Prefix restore changed bits or untouched tails");
  reject([&]{gpu.WritePrefixGroups(target,groups,capacity,width,64,std::vector<float>(1));});
  reject([&]{gpu.ReadPrefixGroups(source,groups,capacity,width,capacity+1);});
  reject([&]{gpu.ReadPrefixGroups(source,0,capacity,width,64);});
  reject([&]{gpu.ReadPrefixGroups(source,groups,capacity,0,64);});
  reject([&]{gpu.ReadPrefixGroups(source,groups,SIZE_MAX,2,64);});
  reject([&]{gpu.WritePrefixGroups(target,groups,capacity,width+1,64,prefix);});
  actual=gpu.Download(target);need(!std::memcmp(actual.data(),full.data(),full.size()*4),"Rejected copy mutated state");
  gpu.Begin();reject([&]{gpu.ReadPrefixGroups(source,groups,capacity,width,64);});reject([&]{gpu.WritePrefixGroups(target,groups,capacity,width,64,prefix);});gpu.Abort();
  auto out=gpu.Allocate(rows,columns);gpu.Begin();gpu.EncodeAdd(source,source,out);gpu.Abort();reject([&]{gpu.ReadPrefixGroups(out,groups,capacity,width,64);});
  gpu.Zero(out);auto zeros=gpu.ReadPrefixGroups(out,groups,capacity,width,64);for(float v:zeros)need(v==0,"Aborted copy did not recover after zero");
  gpu.Begin();gpu.EncodeAdd(source,source,out);gpu.SubmitAndWait();auto doubled=gpu.ReadPrefixGroups(out,groups,capacity,width,64);for(size_t i=0;i<expected.size();++i)need(doubled[i]==2*expected[i],"Completed GPU producer prefix mismatch");++checks;
 }
 reject([&]{gpu.ReadPrefixGroups(Garnet::MetalMatrix(),1,64,1,64);});
 std::cout<<"Native prefix copy bits, row strides, tail preservation, rejection atomicity, pending/abort/recovery PASS "<<checks<<" cases\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
