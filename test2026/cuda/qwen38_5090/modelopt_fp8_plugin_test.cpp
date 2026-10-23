// SPDX-License-Identifier: Apache-2.0
#include "modelopt_fp8_plugin.h"
#include "modelopt_fp8.cuh"
#include <NvInfer.h>
#include <cuda_bf16.h>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>
struct Logger: nvinfer1::ILogger {void log(Severity s,const char* m) noexcept override {if(s<=Severity::kWARNING)fprintf(stderr,"TRT: %s\n",m);}};
void CUDA(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> std::unique_ptr<T> Own(T* p){if(!p)throw std::runtime_error("TensorRT object creation failed");return std::unique_ptr<T>(p);}
float Decode(unsigned char code){
    const int e=(code>>3)&15,m=code&7;
    const float v=e?std::ldexp(1.f+m/8.f,e-7):std::ldexp(float(m),-9);
    return code&128?-v:v;
}
unsigned char Encode(float value){
    const bool negative=std::signbit(value);value=std::fabs(value);
    unsigned best=0;float distance=std::numeric_limits<float>::infinity();
    for(unsigned code=0;code<127;++code){const float delta=std::fabs(Decode(code)-value);
        if(delta<distance||(delta==distance&&!(code&1))){distance=delta;best=code;}}
    return static_cast<unsigned char>(best|(negative?128:0));
}
int main(){try{
    using namespace nvinfer1;Logger logger;
    if(!Garnet::EnsureModelOptFP8PluginRegistered())throw std::runtime_error("Plugin registration failed");
    {
        Garnet::ModelOptFP8Plugin valid(256,256,1.f,1.f/448.f);
        std::vector<unsigned char> wire(valid.getSerializationSize());valid.serialize(wire.data());
        if(Garnet::ModelOptFP8Plugin::Deserialize(wire.data(),wire.size()-1))throw std::runtime_error("Truncated plugin metadata accepted");
        const float invalid=std::numeric_limits<float>::quiet_NaN();std::memcpy(wire.data()+8,&invalid,4);
        if(Garnet::ModelOptFP8Plugin::Deserialize(wire.data(),wire.size()))throw std::runtime_error("Nonfinite plugin metadata accepted");
    }
    constexpr int rows=256,k=256,blocks=k/16;
    for(int batch:{1,16,4096}){
        auto builder=Own(createInferBuilder(logger));
        auto network=Own(builder->createNetworkV2(1U<<int(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        auto config=Own(builder->createBuilderConfig());config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,64ULL<<20);
        std::vector<unsigned char> weights(rows*k);
        const float activations[]={0,.125f,-1.1f,2.25f,450.f,-450.f,.001f,-.001f};
        for(int r=0;r<rows;++r)for(int c=0;c<k;++c)weights[r*k+c]=Encode((1+(r+c)%4)*((r+c)%3?-1.f:1.f));
        std::vector<float> decodedWeights(weights.size());
        for(size_t i=0;i<weights.size();++i)decodedWeights[i]=Decode(weights[i]);
        auto* w=network->addConstant(Dims2{rows,k},{DataType::kINT8,weights.data(),int64_t(weights.size())});
        auto* x=network->addInput("input",DataType::kBF16,Dims2{batch,k});
        if(!w||!x)throw std::runtime_error("Native plugin input creation failed");
        auto plugin=std::make_unique<Garnet::ModelOptFP8Plugin>(rows,k,.5f,1.f);
        ITensor* inputs[]={x,w->getOutput(0)};
        auto* layer=network->addPluginV2(inputs,2,*plugin);
        if(!layer)throw std::runtime_error("Native plugin layer creation failed");
        layer->getOutput(0)->setName("output");network->markOutput(*layer->getOutput(0));
        auto plan=Own(builder->buildSerializedNetwork(*network,*config));
        auto runtime=Own(createInferRuntime(logger));auto engine=Own(runtime->deserializeCudaEngine(plan->data(),plan->size()));
        auto context=Own(engine->createExecutionContext());
        std::vector<__nv_bfloat16> input(batch*k),output(batch*rows);
        for(int b=0;b<batch;++b)for(int c=0;c<k;++c)input[b*k+c]=__float2bfloat16(activations[(b+c)%8]);
        std::vector<float> quantized(input.size());
        for(size_t i=0;i<input.size();++i)quantized[i]=Decode(Encode(__bfloat162float(input[i])));
        void *deviceInput=nullptr,*deviceOutput=nullptr;
        CUDA(cudaMalloc(&deviceInput,input.size()*2));CUDA(cudaMalloc(&deviceOutput,output.size()*2));
        CUDA(cudaMemcpy(deviceInput,input.data(),input.size()*2,cudaMemcpyHostToDevice));
        if(!context->setTensorAddress("input",deviceInput)||!context->setTensorAddress("output",deviceOutput))throw std::runtime_error("Binding failed");
        cudaStream_t stream=nullptr;CUDA(cudaStreamCreate(&stream));
        if(!context->enqueueV3(stream))throw std::runtime_error("Native plugin enqueue failed");CUDA(cudaStreamSynchronize(stream));
        auto verify=[&]{
            CUDA(cudaMemcpy(output.data(),deviceOutput,output.size()*2,cudaMemcpyDeviceToHost));
            for(int b=0;b<batch;++b)for(int r=0;r<rows;++r){
                float expected=0;for(int c=0;c<k;++c)expected+=quantized[b*k+c]*decodedWeights[r*k+c]*.5f;
                if(!std::isfinite(__bfloat162float(output[b*rows+r]))||std::fabs(__bfloat162float(output[b*rows+r])-__bfloat162float(__float2bfloat16(expected)))>.0625f+std::fabs(expected)*.01f)throw std::runtime_error("Serialized native plugin output differs from CPU oracle");
            }
        };
        verify();
        CUDA(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        if(!context->enqueueV3(stream))throw std::runtime_error("Captured plugin enqueue failed");
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        CUDA(cudaStreamEndCapture(stream,&graph));CUDA(cudaGraphInstantiateWithFlags(&executable,graph,0));
        CUDA(cudaGraphLaunch(executable,stream));CUDA(cudaStreamSynchronize(stream));verify();
        CUDA(cudaGraphExecDestroy(executable));CUDA(cudaGraphDestroy(graph));CUDA(cudaStreamDestroy(stream));
        CUDA(cudaFree(deviceInput));CUDA(cudaFree(deviceOutput));
        printf("FP8_PLUGIN_SERIALIZATION_AND_GRAPH_PASS: batch=%d, CPU-oracle outputs=%d; no performance claim\n",batch,batch*rows);fflush(stdout);
    }
    return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
