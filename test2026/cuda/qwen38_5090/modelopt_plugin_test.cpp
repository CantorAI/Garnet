// SPDX-License-Identifier: Apache-2.0
#include "modelopt_nvfp4_plugin.h"
#include "modelopt_nvfp4.cuh"
#include <NvInfer.h>
#include <cuda_bf16.h>
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
struct PluginDestroy{void operator()(nvinfer1::IPluginV2* p) const noexcept{if(p)p->destroy();}};
template<class T> std::unique_ptr<T,PluginDestroy> OwnPlugin(T* p){if(!p)throw std::runtime_error("TensorRT plugin creation failed");return std::unique_ptr<T,PluginDestroy>(p);}
size_t Offset(int row,int col,int columns){return size_t(row/128)*128*columns+size_t(col/4)*512+(row%32)*16+((row%128)/32)*4+col%4;}
int main(int argc,char** argv){try{
    using namespace nvinfer1;Logger logger;
    if(argc==2&&!std::strcmp(argv[1],"--candidate-contracts")){
        if(!Garnet::EnsureModelOptNVFP4PluginRegistered())throw std::runtime_error("Dual plugin registration failed");
        Garnet::ModelOptNVFP4Plugin legacy(5120,17408,1.f,1.f),candidate(5120,17408,1.f,1.f,true);
        if(std::strcmp(legacy.getPluginVersion(),"1")||legacy.getSerializationSize()!=16||std::strcmp(candidate.getPluginVersion(),"2")||candidate.getSerializationSize()!=20)throw std::runtime_error("Plugin wire/version contract changed");
        std::vector<unsigned char> wire(20);candidate.serialize(wire.data());
        std::unique_ptr<Garnet::ModelOptNVFP4Plugin> restored(Garnet::ModelOptNVFP4Plugin::Deserialize(wire.data(),wire.size()));
        auto clone=OwnPlugin(candidate.clone());
        if(!restored||std::strcmp(restored->getPluginVersion(),"2")||std::strcmp(clone->getPluginVersion(),"2"))throw std::runtime_error("Candidate policy lost on clone/reload");
        std::vector<unsigned char> again(20);restored->serialize(again.data());if(again!=wire)throw std::runtime_error("Candidate serialization not stable");
        for(int batch:{1,2,4,128}){
            PluginTensorDesc in[3]{};in[0].dims=Dims2{batch,17408};
            const auto a=legacy.getWorkspaceSize(in,3,nullptr,1),b=candidate.getWorkspaceSize(in,3,nullptr,1);
            if(!a||b-a!=((batch==2)?0:60ULL<<20)||restored->getWorkspaceSize(in,3,nullptr,1)!=b||clone->getWorkspaceSize(in,3,nullptr,1)!=b)throw std::runtime_error("Serialized workspace policy mismatch");
        }
        auto* oldCreator=getPluginRegistry()->getPluginCreator("GarnetModelOptNVFP4","1","");auto* newCreator=getPluginRegistry()->getPluginCreator("GarnetModelOptNVFP4","2","");
        if(!oldCreator||!newCreator||oldCreator->deserializePlugin("wrong",wire.data(),20))throw std::runtime_error("Legacy creator admitted candidate wire");
        std::vector<unsigned char> oldWire(16);legacy.serialize(oldWire.data());
        if(newCreator->deserializePlugin("wrong",oldWire.data(),16))throw std::runtime_error("Candidate creator admitted legacy wire");
        auto old=OwnPlugin(oldCreator->deserializePlugin("legacy",oldWire.data(),16));auto current=OwnPlugin(newCreator->deserializePlugin("candidate",wire.data(),20));
        if(std::strcmp(old->getPluginVersion(),"1")||std::strcmp(current->getPluginVersion(),"2"))throw std::runtime_error("Creator changed serialized selection policy");
        if(Garnet::ModelOptNVFP4Plugin::Deserialize(wire.data(),19))throw std::runtime_error("Truncated candidate wire admitted");
        std::fill(wire.begin()+16,wire.end(),0);if(Garnet::ModelOptNVFP4Plugin::Deserialize(wire.data(),20))throw std::runtime_error("Unknown candidate policy admitted");
        bool rejected=false;try{Garnet::ModelOptNVFP4Plugin invalid(256,256,1.f,1.f,true);}catch(const std::invalid_argument&){rejected=true;}
        if(!rejected)throw std::runtime_error("Candidate admitted unqualified geometry");
        puts("NVFP4_CANDIDATE_SERIALIZATION_CLONE_WORKSPACE_LEGACY_CONTRACTS_PASS; no GPU timing claim");return 0;
    }
    {
        Garnet::ModelOptNVFP4Plugin plugin(256,256,1.f,1.f/448.f);
        PluginTensorDesc desc[4]{};
        for(auto& d:desc)d.format=TensorFormat::kLINEAR;
        desc[0].type=desc[3].type=DataType::kBF16;
        for(int mask=0;mask<(1<<2);++mask){
            for(int i=1;i<3;++i)desc[i].type=(mask&(1<<(i-1)))?DataType::kINT32:DataType::kINT8;
            for(int pos=0;pos<4;++pos)if(!plugin.supportsFormatCombination(pos,desc,3,1))throw std::runtime_error("Packed storage type rejected");
        }
        for(int pos=0;pos<4;++pos){
            auto saved=desc[pos];desc[pos].type=DataType::kFLOAT;
            if(plugin.supportsFormatCombination(pos,desc,3,1))throw std::runtime_error("Wrong storage type admitted");
            desc[pos]=saved;desc[pos].format=TensorFormat::kCHW4;
            if(plugin.supportsFormatCombination(pos,desc,3,1))throw std::runtime_error("Nonlinear packed storage admitted");
            desc[pos]=saved;
        }
        if(plugin.supportsFormatCombination(0,nullptr,3,1)||plugin.supportsFormatCombination(-1,desc,3,1)||plugin.supportsFormatCombination(4,desc,3,1))throw std::runtime_error("Invalid descriptor admitted");
        puts("COMPRESSED_STORAGE_FORMAT_ADMISSION_PASS; no GPU or numerical claim");
        if(argc==2&&!std::strcmp(argv[1],"--formats-only"))return 0;
    }
    if(!Garnet::EnsureModelOptNVFP4PluginRegistered())throw std::runtime_error("Plugin registration failed");
    {
        Garnet::ModelOptNVFP4Plugin valid(256,256,1.f,1.f/448.f);
        std::vector<unsigned char> wire(valid.getSerializationSize());valid.serialize(wire.data());
        if(Garnet::ModelOptNVFP4Plugin::Deserialize(wire.data(),wire.size()-1))throw std::runtime_error("Truncated plugin metadata accepted");
        const float invalid=std::numeric_limits<float>::quiet_NaN();std::memcpy(wire.data()+8,&invalid,4);
        if(Garnet::ModelOptNVFP4Plugin::Deserialize(wire.data(),wire.size()))throw std::runtime_error("Nonfinite plugin metadata accepted");
    }
    constexpr int rows=256,k=256,blocks=k/16;
    for(bool external:{false,true})for(int batch:{1,16}){
        auto builder=Own(createInferBuilder(logger));
        auto network=Own(builder->createNetworkV2(1U<<int(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
        auto config=Own(builder->createBuilderConfig());config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,64ULL<<20);
        std::vector<unsigned char> weights(rows*k/2,0x42),scales(Garnet::ModelOptNVFP4ScaleBytes(rows,k));
        const unsigned char codes[]={56,64,72};const float values[]={1,2,4},activations[]={0,.5f,1.f,1.5f};
        for(int r=0;r<rows;++r)for(int b=0;b<blocks;++b)scales[Offset(r,b,blocks)]=codes[(r+b)%3];
        auto* w=external?nullptr:network->addConstant(Dims2{rows,k/2},{DataType::kINT8,weights.data(),int64_t(weights.size())});
        Dims scaleShape{};scaleShape.nbDims=1;scaleShape.d[0]=int64_t(scales.size());
        auto* s=external?nullptr:network->addConstant(scaleShape,{DataType::kINT8,scales.data(),int64_t(scales.size())});
        auto* weightTensor=external?network->addInput("packed_weight",DataType::kINT32,Dims2{rows,k/8}):(w?w->getOutput(0):nullptr);
        scaleShape.d[0]/=4;
        auto* scaleTensor=external?network->addInput("packed_scales",DataType::kINT32,scaleShape):(s?s->getOutput(0):nullptr);
        auto* x=network->addInput("input",DataType::kBF16,Dims2{batch,k});
        if(!weightTensor||!scaleTensor||!x)throw std::runtime_error("Native plugin input creation failed");
        auto plugin=std::make_unique<Garnet::ModelOptNVFP4Plugin>(rows,k,.5f,1.f/448.f);
        ITensor* inputs[]={x,weightTensor,scaleTensor};
        auto* layer=network->addPluginV2(inputs,3,*plugin);
        if(!layer)throw std::runtime_error("Native plugin layer creation failed");
        layer->getOutput(0)->setName("output");network->markOutput(*layer->getOutput(0));
        auto plan=Own(builder->buildSerializedNetwork(*network,*config));
        auto runtime=Own(createInferRuntime(logger));auto engine=Own(runtime->deserializeCudaEngine(plan->data(),plan->size()));
        auto context=Own(engine->createExecutionContext());
        std::vector<__nv_bfloat16> input(batch*k),output(batch*rows);
        for(int b=0;b<batch;++b)for(int c=0;c<k;++c)input[b*k+c]=__float2bfloat16(activations[(b+c)%4]);
        void *deviceInput=nullptr,*deviceOutput=nullptr;
        CUDA(cudaMalloc(&deviceInput,input.size()*2));CUDA(cudaMalloc(&deviceOutput,output.size()*2));
        CUDA(cudaMemcpy(deviceInput,input.data(),input.size()*2,cudaMemcpyHostToDevice));
        if(!context->setTensorAddress("input",deviceInput)||!context->setTensorAddress("output",deviceOutput))throw std::runtime_error("Binding failed");
        void *deviceWeight=nullptr,*deviceScales=nullptr;
        if(external){
            CUDA(cudaMalloc(&deviceWeight,weights.size()));
            CUDA(cudaMemcpy(deviceWeight,weights.data(),weights.size(),cudaMemcpyHostToDevice));
            if(!context->setTensorAddress("packed_weight",deviceWeight))throw std::runtime_error("External packed weight binding failed");
            CUDA(cudaMalloc(&deviceScales,scales.size()));
            CUDA(cudaMemcpy(deviceScales,scales.data(),scales.size(),cudaMemcpyHostToDevice));
            if(!context->setTensorAddress("packed_scales",deviceScales))throw std::runtime_error("External scale binding failed");
        }
        cudaStream_t stream=nullptr;CUDA(cudaStreamCreate(&stream));
        if(!context->enqueueV3(stream))throw std::runtime_error("Native plugin enqueue failed");CUDA(cudaStreamSynchronize(stream));
        auto verify=[&]{
            CUDA(cudaMemcpy(output.data(),deviceOutput,output.size()*2,cudaMemcpyDeviceToHost));
            for(int b=0;b<batch;++b)for(int r=0;r<rows;++r){
                float expected=0;for(int c=0;c<k;++c)expected+=__bfloat162float(input[b*k+c])*(c%2?2.f:1.f)*values[(r+c/16)%3]*.5f;
                if(__bfloat162float(output[b*rows+r])!=__bfloat162float(__float2bfloat16(expected)))throw std::runtime_error("Serialized native plugin output differs from CPU oracle");
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
        if(deviceWeight)CUDA(cudaFree(deviceWeight));if(deviceScales)CUDA(cudaFree(deviceScales));
        printf("NVFP4_PLUGIN_SERIALIZATION_AND_GRAPH_PASS: external_int32=%d, batch=%d, exact outputs=%d; no performance claim\n",int(external),batch,batch*rows);fflush(stdout);
    }
    return 0;
}catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}}
