#include "metal_context.h"
#include <cstring>
#include <cmath>
#include <iostream>
#include <stdexcept>
int main(){try{Garnet::MetalContext gpu;size_t checks=0;
for(size_t rows:{1,2})for(size_t inner:{2048,2560,9216})for(size_t columns:{257,272,511,768,784,1024}){
 std::vector<float> input(rows*inner),weight(columns*inner);std::vector<uint16_t> bits(weight.size());
 for(size_t i=0;i<input.size();++i)input[i]=float(int((i*7+i/inner)%17)-8)/32;
 for(size_t i=0;i<weight.size();++i){weight[i]=float(int((i*11+i/inner)%19)-9)/64;uint32_t raw;std::memcpy(&raw,&weight[i],4);bits[i]=uint16_t(raw>>16);}
 auto a=gpu.Upload(input,rows,inner),out=gpu.Allocate(rows,columns);auto w=gpu.UploadBF16(bits.data(),bits.size()*2,columns,inner);
 if(!gpu.PrepareBF16BodyShape(a,w,out))throw std::runtime_error("Specialized shape unavailable");
 std::vector<float> control;
 for(bool enabled:{false,true}){Garnet::MetalContext::BenchmarkDecode256(enabled);auto before=Garnet::MetalContext::BenchmarkDecode256Dispatches();gpu.Begin();gpu.EncodeBF16MatVec(a,w,out);gpu.SubmitAndWait();auto actual=gpu.Download(out);
 bool routed=enabled&&rows==1&&(inner==2560||inner==9216)&&(columns%32==0||columns%16!=0);if(Garnet::MetalContext::BenchmarkDecode256Dispatches()-before!=size_t(routed))throw std::runtime_error("Wrong decode256 route");
 if(!enabled)control=actual;else if(std::memcmp(control.data(),actual.data(),actual.size()*4))throw std::runtime_error("Decode grouping changed output bits");
 for(size_t row=0;row<rows;++row)for(size_t col=0;col<columns;++col){double ref=0;for(size_t k=0;k<inner;++k)ref+=double(input[row*inner+k])*weight[col*inner+k];if(!std::isfinite(actual[row*columns+col])||std::abs(actual[row*columns+col]-ref)>1e-5)throw std::runtime_error("FP64 reference mismatch");}
 ++checks;}
}Garnet::MetalContext::BenchmarkDecode256(false);std::cout<<"Decode256 exact bits, FP64 bounds, tails, row and reasoner fallback PASS "<<checks<<" policies\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
