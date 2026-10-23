// SPDX-License-Identifier: Apache-2.0
#include "modelopt_trt_projection.h"
#include <NvInferRuntime.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>
struct Logger: nvinfer1::ILogger {
    void log(Severity s,const char* m) noexcept override {if(s<=Severity::kWARNING)fprintf(stderr,"TRT: %s\n",m);}
};
void Check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
template<class T> std::unique_ptr<T> Own(T* x){if(!x)throw std::runtime_error("TensorRT object creation failed");return std::unique_ptr<T>(x);}
int main(int argc,char** argv){
    try{
        using namespace nvinfer1;Logger logger;
        constexpr int m=16,n=128,k=128;
        for(bool nvfp4:{true,false}){
            auto builder=Own(createInferBuilder(logger));
            auto net=Own(builder->createNetworkV2(1U<<int(NetworkDefinitionCreationFlag::kSTRONGLY_TYPED)));
            auto cfg=Own(builder->createBuilderConfig());cfg->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE,64ULL<<20);
            cfg->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
            std::vector<unsigned char> weight(n*k/(nvfp4?2:1),nvfp4?0x22:0x38),scales(n*k/16,0x38);
            const float scale2=.5f,inputScale=nvfp4?1.f/448.f:.25f,fp8Scale=.5f;
            auto projection=Garnet::AddModelOptTRTWeight(*net,Dims2{n,k},
                {nvfp4?DataType::kFP4:DataType::kFP8,weight.data(),n*k},
                nvfp4?Weights{DataType::kFP8,scales.data(),n*k/16}:Weights{DataType::kFLOAT,&fp8Scale,1},
                {DataType::kFLOAT,&scale2,1},{DataType::kFLOAT,&inputScale,1},nvfp4);
            auto* x=net->addInput("input",DataType::kBF16,Dims2{m,k});
            if(!x)throw std::runtime_error("Input creation failed");
            auto* y=Garnet::ProjectModelOptTRT(*net,*x,projection);y->setName("output");net->markOutput(*y);
            auto plan=Own(builder->buildSerializedNetwork(*net,*cfg));
            auto runtime=Own(createInferRuntime(logger));auto engine=Own(runtime->deserializeCudaEngine(plan->data(),plan->size()));
            auto context=Own(engine->createExecutionContext());
            std::vector<__nv_bfloat16> input(m*k),output(m*n);
            const float values[]={0,.25f,.5f,.75f,1.f,1.5f};
            for(int i=0;i<m*k;++i)input[i]=__float2bfloat16(values[i%6]);
            void *deviceInput=nullptr,*deviceOutput=nullptr;
            Check(cudaMalloc(&deviceInput,input.size()*2));Check(cudaMalloc(&deviceOutput,output.size()*2));
            Check(cudaMemcpy(deviceInput,input.data(),input.size()*2,cudaMemcpyHostToDevice));
            if(!context->setTensorAddress("input",deviceInput)||!context->setTensorAddress("output",deviceOutput)||!context->enqueueV3(nullptr))
                throw std::runtime_error("ModelOpt engine enqueue failed");
            Check(cudaMemcpy(output.data(),deviceOutput,output.size()*2,cudaMemcpyDeviceToHost));
            for(int row=0;row<m;++row){
                float expected=0;for(int col=0;col<k;++col)expected+=__bfloat162float(input[row*k+col])*.5f;
                expected=__bfloat162float(__float2bfloat16(expected));
                for(int col=0;col<n;++col)if(__bfloat162float(output[row*n+col])!=expected)
                    throw std::runtime_error("TensorRT quantized projection differs from exact oracle");
            }
            if(argc==2){auto inspector=Own(engine->createEngineInspector());
                std::ofstream file(std::string(argv[1])+(nvfp4?"-nvfp4.json":"-fp8.json"));
                file<<inspector->getEngineInformation(LayerInformationFormat::kJSON);
            }
            Check(cudaFree(deviceInput));Check(cudaFree(deviceOutput));
            printf("MODELOPT_TRT_%s_PASS: %d exact outputs; serialized bytes=%zu; no performance claim\n",nvfp4?"NVFP4_W4A4":"FP8_W8A8",m*n,plan->size());
        }
        return 0;
    }catch(const std::exception& e){fprintf(stderr,"%s\n",e.what());return 1;}
}
