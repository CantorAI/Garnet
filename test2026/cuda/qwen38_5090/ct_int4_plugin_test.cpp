// SPDX-License-Identifier: Apache-2.0
// Bounded compatibility diagnostic. No serving admission or performance claim.
#include <NvInfer.h>
#include "ct_int4_plugin.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <algorithm>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
struct Logger: nvinfer1::ILogger {
 void log(Severity s,const char* m) noexcept override {if(s<=Severity::kWARNING)fprintf(stderr,"TRT: %s\n",m);}
};
template<class T> std::unique_ptr<T> Own(T* p){if(!p)throw std::runtime_error("TensorRT creation failed");return std::unique_ptr<T>(p);}
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct Buffer {void* p=nullptr; explicit Buffer(size_t n){Check(cudaMalloc(&p,n));} ~Buffer(){if(p)cudaFree(p);}};
int main(int argc,char** argv){try{
 using namespace nvinfer1;
 if(argc!=2)throw std::runtime_error("Output prefix and optional kn layout required");
 const bool model=false;
 const bool kn=false;
 if(argc==3&&!kn)throw std::runtime_error("Unknown physical layout");
 if(!Garnet::EnsureCTInt4G128PluginRegistered())throw std::runtime_error("Plugin registration failed");
 Logger logger;const int n=model?5120:128,k=model?6144:256,group=128;
 // Independent logical oracle spans every signed code and distinct row/group scales.
 std::vector<int> logical(n*k);std::vector<unsigned char> offset(n*k/2,0),packed(n*k/2);
 std::vector<__nv_bfloat16> scales(n*k/group);
 for(int row=0;row<n;++row){
  for(int g=0;g<k/group;++g)scales[row*(k/group)+g]=__float2bfloat16((1+(row+g)%4)/16.f);
  for(int col=0;col<k;++col){int q=(row*7+col*3+col/16)%16-8;logical[row*k+col]=q;
   int i=row*k+col;offset[i/2]|=static_cast<unsigned char>((q+8)<<((i%2)*4));}
 }
 packed=offset;
 std::vector<__nv_bfloat16> storedScales=scales;
 if(kn){
  std::fill(packed.begin(),packed.end(),0);
  for(int j=0;j<k;++j)for(int row=0;row<n;++row){int i=j*n+row;unsigned q=static_cast<unsigned>(logical[row*k+j])&15U;packed[i/2]|=static_cast<unsigned char>(q<<((i%2)*4));}
  for(int g=0;g<k/group;++g)for(int row=0;row<n;++row)storedScales[g*n+row]=scales[row*(k/group)+g];
 }
 for(int m:{1,16}){
  auto builder=Own(createInferBuilder(logger));
  auto net=Own(builder->createNetworkV2(1U<<int(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
  auto config=Own(builder->createBuilderConfig());config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,64ULL<<20);
  config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
  auto* w=net->addConstant(Dims2{n,k/2},Weights{DataType::kINT8,packed.data(),n*k/2});
  auto* s=net->addConstant(kn?Dims2{k/group,n}:Dims2{n,k/group},Weights{DataType::kBF16,storedScales.data(),n*k/group});
  auto* x=net->addInput("input",DataType::kBF16,Dims2{m,k});
  if(!w||!s||!x)throw std::runtime_error("INT4/BF16 graph input creation failed");
  Garnet::CTInt4G128Plugin plugin(n,k);
  ITensor* inputs[]={x,w->getOutput(0),s->getOutput(0)};
  auto* layer=net->addPluginV2(inputs,3,plugin);
  if(!layer)throw std::runtime_error("Native INT4 plugin graph creation failed");
  layer->setName("native_ct_int4_g128");auto* y=layer->getOutput(0);y->setName("output");net->markOutput(*y);
  auto plan=Own(builder->buildSerializedNetwork(*net,*config));
  auto runtime=Own(createInferRuntime(logger));auto engine=Own(runtime->deserializeCudaEngine(plan->data(),plan->size()));
  auto context=Own(engine->createExecutionContext());auto inspector=Own(engine->createEngineInspector());
  std::ofstream file(std::string(argv[1])+"-m"+std::to_string(m)+".json");
  file<<inspector->getEngineInformation(LayerInformationFormat::kJSON);file.close();
  if(!file)throw std::runtime_error("Cannot retain inspector evidence");
  std::vector<__nv_bfloat16> input(m*k),output(m*n);
  for(int i=0;i<m*k;++i)input[i]=__float2bfloat16(((i*5+i/k)%7-3)/2.f);
  Buffer di(input.size()*2),dout(output.size()*2);
  Check(cudaMemcpy(di.p,input.data(),input.size()*2,cudaMemcpyHostToDevice));
  if(!context->setTensorAddress("input",di.p)||!context->setTensorAddress("output",dout.p)||!context->enqueueV3(nullptr))throw std::runtime_error("INT4 enqueue failed");
  Check(cudaMemcpy(output.data(),dout.p,output.size()*2,cudaMemcpyDeviceToHost));
  for(int row=0;row<m;++row)for(int col=0;col<n;++col){
   float sum=0;for(int j=0;j<k;++j)sum+=__bfloat162float(input[row*k+j])*logical[col*k+j]*__bfloat162float(scales[col*(k/group)+j/group]);
   float expected=__bfloat162float(__float2bfloat16(sum)),actual=__bfloat162float(output[row*n+col]);
   if(actual!=expected){fprintf(stderr,"m=%d row=%d col=%d actual=%g expected=%g\n",m,row,col,actual,expected);throw std::runtime_error("Signed INT4 group128 exact oracle mismatch");}
  }
  printf("NATIVE_INT4_PLUGIN_ARITHMETIC_PASS m=%d outputs=%d plan_bytes=%zu; direct packed CUDA plugin; not full model; no benchmark\n",m,m*n,plan->size());
 }
 return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
