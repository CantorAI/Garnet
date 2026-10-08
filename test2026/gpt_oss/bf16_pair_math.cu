// SPDX-License-Identifier: Apache-2.0
// Complete saved matrices; correctness only, no transport or timing claims.
#include "bf16_pair_math.cuh"
#include "gpt_oss_kernels.h"
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>
#define CUDA_OK(x) do{auto e=(x);if(e!=cudaSuccess){std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e));std::exit(1);}}while(0)
uint32_t bits(float f){uint32_t b;std::memcpy(&b,&f,4);return b;}
float value(uint32_t b){float f;std::memcpy(&f,&b,4);return f;}
float roundBf16(float f){uint32_t b=bits(f);return value((b+0x7fffU+((b>>16)&1U))&0xffff0000U);}
uint16_t finite(uint16_t b){return (b&0x7f80U)==0x7f80U?uint16_t(b^0x80U):b;}
std::array<float,2> inputs(size_t i,int family,int offset){
    uint16_t a=finite(uint16_t(i+offset)),b=0;
    switch(family){
      case 0:b=a;break;
      case 1:b=uint16_t(a^0x8000U);break;
      case 2:b=finite(uint16_t(a+1));break;
      case 3:b=uint16_t(a&0x8000U);break;
      case 4:b=uint16_t((a&0x8000U)|1U);break;
      default:b=uint16_t((a&0x8000U)|0x3f80U);break;
    }
    return {value(uint32_t(a)<<16),value(uint32_t(b)<<16)};
}
void save(const std::filesystem::path& path,const std::vector<float>& data){
    if(std::filesystem::exists(path))std::exit(5);
    FILE* f=std::fopen(path.string().c_str(),"wb");if(!f)std::exit(5);
    if(std::fwrite(data.data(),4,data.size(),f)!=data.size()||std::fclose(f))std::exit(5);
}
int main(int argc,char** argv){
    if(argc!=3)return 2;const int device=std::atoi(argv[1]);const auto folder=std::filesystem::path(argv[2]);
    if(!std::filesystem::is_directory(folder))return 2;
    int original=0;CUDA_OK(cudaGetDevice(&original));CUDA_OK(cudaSetDevice(device));
    cudaDeviceProp prop{};CUDA_OK(cudaGetDeviceProperties(&prop,device));
    std::printf("CORRECTNESS_ONLY device=%d name=%s SM=%d%d NO_PEER_NO_NCCL_NO_TIMING\n",device,prop.name,prop.major,prop.minor);
    size_t valuesChecked=0;
    for(int rows:{224,512,3584,4096}){
      const size_t count=size_t(rows)*2880;
      std::array<float*,2> gpu{};std::array<void*,2> packed{};float* output=nullptr;cudaStream_t stream{};
      CUDA_OK(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking));
      for(int side=0;side<2;++side){CUDA_OK(cudaMalloc(&gpu[side],count*4));CUDA_OK(cudaMalloc(&packed[side],count*2));}
      CUDA_OK(cudaMalloc(&output,count*4));
      auto reject=[&](const void* a,const void* b,float* o,size_t n){if(GarnetPrototype::pairMath(a,b,o,n,stream)!=cudaErrorInvalidValue)std::exit(4);};
      reject(nullptr,packed[1],output,count);reject(packed[0],nullptr,output,count);reject(packed[0],packed[1],nullptr,count);
      for(size_t n:{size_t(0),size_t(7),GarnetPrototype::kMaximumPairElements+8,SIZE_MAX})reject(packed[0],packed[1],output,n);
      reject(static_cast<char*>(packed[0])+2,packed[1],output,count);reject(packed[0],static_cast<char*>(packed[1])+2,output,count);
      reject(packed[0],packed[1],output+1,count);
      auto execute=[&]{for(int side=0;side<2;++side)CUDA_OK(Garnet::GptOssTpPackBf16(gpu[side],packed[side],count,stream));CUDA_OK(GarnetPrototype::pairMath(packed[0],packed[1],output,count,stream));};
      cudaGraph_t graph{};cudaGraphExec_t execution{};
      CUDA_OK(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));execute();CUDA_OK(cudaStreamEndCapture(stream,&graph));CUDA_OK(cudaGraphInstantiate(&execution,graph,0));
      std::array<std::vector<float>,2> host{std::vector<float>(count),std::vector<float>(count)};std::vector<float> actual(count),expected(count);
      for(int family=0;family<6;++family)for(int mode=0;mode<2;++mode){
        const int offset=mode?131:0;
        for(size_t i=0;i<count;++i){auto pair=inputs(i,family,offset);host[0][i]=pair[0];host[1][i]=pair[1];volatile float sum=pair[0]+pair[1];expected[i]=roundBf16(sum);}
        for(int side=0;side<2;++side)CUDA_OK(cudaMemcpyAsync(gpu[side],host[side].data(),count*4,cudaMemcpyHostToDevice,stream));
        if(mode)for(int replay=0;replay<3;++replay)CUDA_OK(cudaGraphLaunch(execution,stream));else execute();
        CUDA_OK(cudaMemcpyAsync(actual.data(),output,count*4,cudaMemcpyDeviceToHost,stream));CUDA_OK(cudaStreamSynchronize(stream));
        const std::string stem="rows"+std::to_string(rows)+"-family"+std::to_string(family)+(mode?"-graph":"-eager");
        // Save whole actual data BEFORE checking, preserving a failure matrix too.
        save(folder/(stem+".bin"),actual);
        for(size_t i=0;i<count;++i)if(bits(actual[i])!=bits(expected[i])){
          std::fprintf(stderr,"FAIL rows=%d family=%d mode=%d i=%zu inputs=%08x/%08x actual=%08x expected=%08x\n",rows,family,mode,i,bits(host[0][i]),bits(host[1][i]),bits(actual[i]),bits(expected[i]));return 3;
        }
        valuesChecked+=count;std::printf("FULL_MATRIX_PASS rows=%d family=%d mode=%s values=%zu offset=%d\n",rows,family,mode?"graph":"eager",count,offset);std::fflush(stdout);
      }
      CUDA_OK(cudaGraphExecDestroy(execution));CUDA_OK(cudaGraphDestroy(graph));
      for(int side=0;side<2;++side){CUDA_OK(cudaFree(gpu[side]));CUDA_OK(cudaFree(packed[side]));}CUDA_OK(cudaFree(output));CUDA_OK(cudaStreamDestroy(stream));
    }
    CUDA_OK(cudaSetDevice(original));std::printf("PAIR_MATH_FULL_BITWISE_PASS matrices=48 values=%zu API_GUARDS40 NO_PEER_NO_NCCL_NO_TIMING\n",valuesChecked);return 0;
}
