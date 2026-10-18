#include <algorithm>
#include "metal_context.h"
#include <cmath>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace Garnet;
static uint64_t Encoded(){return MetalContext::DensePrefetchEncoded();}
static void require(bool v,const char* text){if(!v)throw std::runtime_error(text);}
static float fromBits(uint32_t bits){float f;std::memcpy(&f,&bits,4);return f;}
int main(int argc,char** argv){try{
 MetalContext gpu;std::ofstream binary;if(argc>1)binary.open(argv[1],std::ios::binary);size_t cases=0,passes=0,widePasses=0;double worstRatio=0;
 auto run=[&](size_t rows,size_t inner,size_t columns){
  std::vector<float> input(rows*inner),weights(columns*inner);std::vector<uint16_t> bits(weights.size());
  for(size_t i=0;i<bits.size();++i){int exponent=int((i*13+i/inner)%8)-5;bits[i]=uint16_t(((i*7+i/inner)%2?0x8000:0)|((127+exponent)<<7)|((i*17+i/inner)%128));weights[i]=fromBits(uint32_t(bits[i])<<16);}
  auto a=gpu.Allocate(rows,inner),out=gpu.Allocate(rows,columns);auto w=gpu.UploadBF16(bits.data(),bits.size()*2,columns,inner,false,false,false);bool prepared=gpu.PrepareBF16BodyShape(a,w,out);require(prepared==(rows<=2&&columns<100000),"Unexpected shape preparation admission");
  bool eligible=false;
#ifdef GARNET_METAL_DENSE_PREFETCH
  eligible=rows==1&&(inner==2048||inner==6144)&&columns>256&&columns<100000;
#endif
  require(!w.PrimaryReady(),"Dense prefetch unit requires forced raw storage");
  std::vector<float> saved;
  for(size_t pass=0;pass<2;++pass){
   for(size_t i=0;i<input.size();++i){int exponent=int((i*5+pass*3)%7)-3;uint32_t raw=uint32_t(((i*11+pass)%2?0x80000000u:0)|((127+exponent)<<23)|((i*2654435761u+pass*7919u)&0x7fffffu));input[i]=fromBits(raw);}
   gpu.Update(a,input);gpu.Update(out,std::vector<float>(rows*columns,std::numeric_limits<float>::quiet_NaN()));auto before=Encoded();gpu.Begin();gpu.EncodeBF16MatVec(a,w,out);gpu.SubmitAndWait();auto actual=gpu.Download(out);
   require(Encoded()-before==size_t(eligible),"Incorrect encoded wide route");widePasses+=eligible;require(gpu.Download(a)==input,"Input mutated");
   for(size_t r=0;r<rows;++r)for(size_t c=0;c<columns;++c){double ref=0,l1=0;for(size_t k=0;k<inner;++k){double product=double(input[r*inner+k])*weights[c*inner+k];ref+=product;l1+=std::abs(product);}double bound=rows<=2?1e-4+2e-6*l1:5e-6+5e-5*std::abs(ref);double error=std::abs(double(actual[r*columns+c])-ref);if(!std::isfinite(actual[r*columns+c])||error>bound)std::cerr<<"failure rows="<<rows<<" inner="<<inner<<" columns="<<columns<<" pass="<<pass<<" outputRow="<<r<<" col="<<c<<" actual="<<actual[r*columns+c]<<" ref="<<ref<<" L1="<<l1<<" bound="<<bound<<" error="<<error<<std::endl;require(std::isfinite(actual[r*columns+c])&&error<=bound,"FP64 reference bound failed");worstRatio=std::max(worstRatio,error/bound);}
   if(binary.is_open()){uint64_t shape[4]={rows,inner,columns,pass};binary.write(reinterpret_cast<char*>(shape),sizeof(shape));binary.write(reinterpret_cast<const char*>(actual.data()),actual.size()*4);}
   saved=actual;++passes;
  }
  if(rows==1&&inner==2048&&columns==272){
   auto badout=gpu.Allocate(rows,columns+1);require(!gpu.PrepareBF16BodyShape(a,w,badout),"Invalid output shape prepared");auto invalidBefore=Encoded();bool rejected=false;gpu.Begin();try{gpu.EncodeBF16MatVec(a,w,badout);}catch(...){rejected=true;}gpu.Abort();require(rejected&&Encoded()==invalidBefore,"Invalid encoding reached wide dispatch");
   auto before=Encoded();gpu.Begin();gpu.EncodeBF16MatVec(a,w,out);gpu.EncodeBF16MatVec(a,w,out);w=MetalBF16Weight();gpu.SubmitAndWait();auto value=gpu.Download(out);require(std::memcmp(saved.data(),value.data(),saved.size()*4)==0,"Queued reuse/source release changed result");require(Encoded()-before==2*size_t(eligible),"Queued route count");
  }
  if(rows==1&&inner==6144&&columns==257){
   gpu.Begin();gpu.EncodeBF16MatVec(a,w,out);gpu.Abort();bool rejected=false;try{gpu.Download(out);}catch(...){rejected=true;}require(rejected,"Aborted output readable");gpu.Begin();gpu.EncodeBF16MatVec(a,w,out);gpu.SubmitAndWait();auto value=gpu.Download(out);require(std::memcmp(saved.data(),value.data(),saved.size()*4)==0,"Abort recovery differs");
   gpu.Begin();rejected=false;try{gpu.PrepareBF16BodyShape(a,w,out);}catch(...){rejected=true;}gpu.Abort();require(rejected,"Prepared pipeline during active encoding");
  }
  ++cases;
 };
 for(size_t rows:{1,2})for(size_t inner:{2048,6144,2560,9216,2049,128})for(size_t columns:{257,272,511,768,784,1024})run(rows,inner,columns);
 for(size_t rows:{1,2})for(size_t columns:{16,256})run(rows,2048,columns);
 run(1,128,151936);
 for(size_t rows:{3,16,32,64})for(size_t inner:{2048,2560})for(size_t columns:{272,768})run(rows,inner,columns);
 require(cases==93&&passes==186,"Case coverage mismatch");
#ifdef GARNET_METAL_DENSE_PREFETCH
 require(widePasses>0,"Primary route not executed");
#else
 require(widePasses==0,"Disabled policy routed primary");
#endif
 if(binary.is_open()){binary.flush();require(bool(binary),"Cannot write full bit results");}
 std::cout<<"Forced-raw dense prefetch numerical/reference/poison/input/tail272/fallback/small/vocab/queued owner/Abort recovery PASS cases="<<cases<<" passes="<<passes<<" wide="<<widePasses<<" worst_scaled_error="<<worstRatio<<'\n';return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
