// SPDX-License-Identifier: Apache-2.0
#include "attention_core_trt_probe.h"
#include "../../src/backends/tensorrt/qwen35_attention_core_plugin.h"
#include <NvInfer.h>
#include <cstring>
#include <iostream>
#include <stdexcept>
using namespace nvinfer1;
namespace {
const char* inputs[]={"q","k","v","gate","old_key","old_value","slots","active"};
const char* outputs[]={"context","next_key","next_value"};
struct Logger:ILogger {
    void log(Severity severity,const char* message) noexcept override {
        if(severity<=Severity::kWARNING)std::cerr<<"[TRT] "<<message<<"\n";
    }
};
void require(bool valid,const char* message){if(!valid)throw std::runtime_error(message);}
void checkSerialization(Garnet::Qwen35AttentionCoreConfig c){
    auto* creator=getPluginRegistry()->getPluginCreator("GarnetQwen35AttentionCore","1","");
    require(creator!=nullptr,"Creator missing");
    auto invalid=c;invalid.dim=0;
    require(creator->deserializePlugin("invalid",&invalid,sizeof(invalid))==nullptr,"Invalid config deserialized");
    require(creator->deserializePlugin("truncated",&c,sizeof(c)-1)==nullptr,"Truncated config deserialized");
    auto* plugin=creator->deserializePlugin("roundtrip",&c,sizeof(c));require(plugin!=nullptr,"Valid config rejected");
    plugin->setPluginNamespace("contract-test");
    auto* clone=plugin->clone();require(clone!=nullptr,"Clone failed");
    require(!std::strcmp(clone->getPluginNamespace(),"contract-test"),"Clone lost namespace");
    require(clone->getSerializationSize()==sizeof(c),"Serialization size mismatch");
    Garnet::Qwen35AttentionCoreConfig restored{};clone->serialize(&restored);
    require(!std::memcmp(&c,&restored,sizeof(c)),"Serialization changed config");
    clone->destroy();plugin->destroy();
}
}
struct AttentionCoreTRTProbe::Impl {
    Logger logger;
    std::unique_ptr<IRuntime> runtime;
    std::unique_ptr<ICudaEngine> engine;
    std::unique_ptr<IExecutionContext> context;
};
AttentionCoreTRTProbe::AttentionCoreTRTProbe(Garnet::Qwen35AttentionCoreConfig c):impl_(new Impl){
    require(Garnet::EnsureQwen35AttentionCorePluginRegistered(),"Plugin registration failed");
    checkSerialization(c);
    std::unique_ptr<IBuilder> builder(createInferBuilder(impl_->logger));require(bool(builder),"Builder failed");
    std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(1U<<uint32_t(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
    std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());require(network&&config,"Network/config failed");
    config->clearFlag(BuilderFlag::kTF32);config->setMaxAuxStreams(0);
    config->setBuilderOptimizationLevel(0);config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,16ULL<<20);
    int dimensions[8][4]={{c.batch,c.tokens,c.heads,c.dim},{c.batch,c.tokens,c.kv,c.dim},
        {c.batch,c.tokens,c.kv,c.dim},{c.batch,c.tokens,c.heads,c.dim},
        {c.batch,c.kv,c.capacity,c.dim},{c.batch,c.kv,c.capacity,c.dim},
        {c.batch,c.tokens,0,0},{c.batch,c.tokens,0,0}};
    ITensor* tensors[8];
    for(int i=0;i<8;++i){
        Dims dims{};dims.nbDims=i<6?4:2;
        for(int d=0;d<dims.nbDims;++d)dims.d[d]=dimensions[i][d];
        tensors[i]=network->addInput(inputs[i],DataType::kFLOAT,dims);require(tensors[i]!=nullptr,"Input failed");
    }
    Garnet::Qwen35AttentionCorePlugin plugin(c);
    auto* layer=network->addPluginV2(tensors,8,plugin);require(layer!=nullptr,"Plugin layer failed");
    for(int i=0;i<3;++i){auto* tensor=layer->getOutput(i);require(tensor!=nullptr,"Output missing");tensor->setName(outputs[i]);network->markOutput(*tensor);}
    std::unique_ptr<IHostMemory> plan(builder->buildSerializedNetwork(*network,*config));require(bool(plan),"Plan build failed");
    impl_->runtime.reset(createInferRuntime(impl_->logger));require(bool(impl_->runtime),"Runtime failed");
    impl_->engine.reset(impl_->runtime->deserializeCudaEngine(plan->data(),plan->size()));require(bool(impl_->engine),"Engine deserialize failed");
    impl_->context.reset(impl_->engine->createExecutionContext());require(bool(impl_->context),"Context failed");
}
AttentionCoreTRTProbe::~AttentionCoreTRTProbe()=default;
int AttentionCoreTRTProbe::enqueue(const void* const* in,void* const* out,cudaStream_t stream) noexcept {
    if(!in||!out)return 1;
    for(int i=0;i<8;++i)if(!impl_->context->setTensorAddress(inputs[i],const_cast<void*>(in[i])))return 1;
    for(int i=0;i<3;++i)if(!impl_->context->setTensorAddress(outputs[i],out[i]))return 1;
    return impl_->context->enqueueV3(stream)?0:1;
}
