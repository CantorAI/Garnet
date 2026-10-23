// SPDX-License-Identifier: Apache-2.0
// Real TensorRT context slots, independently controlled CUDA graph lifetimes.
#include "../../src/backends/tensorrt/trt_context_pool.h"
#include <NvInfer.h>
#include <NvInferVersion.h>
#include <cuda_runtime.h>
#include <future>
#include <atomic>
#include <chrono>
#include <iostream>
#include <stdexcept>
#include <thread>
#include <vector>
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
void require(bool b,const char* message){if(!b)throw std::runtime_error(message);}
struct Logger: nvinfer1::ILogger{void log(Severity,const char*) noexcept override{}};
__global__ void spinIncrement(int* value,unsigned long long cycles){if(threadIdx.x||blockIdx.x)return;auto start=clock64();while(clock64()-start<cycles){}++*value;}
void CUDART_CB hostGate(void* pointer){auto& gate=*static_cast<std::atomic<bool>*>(pointer);while(!gate.load(std::memory_order_acquire))std::this_thread::yield();}
void capture(Garnet::TRTContextPool::Slot* slot,cudaStream_t stream,int* value){
    check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));spinIncrement<<<1,1,0,stream>>>(value,1);
    check(cudaStreamEndCapture(stream,&slot->graph));check(cudaGraphInstantiate(&slot->executable,slot->graph,nullptr,nullptr,0));
}
int main(){try{
    check(cudaSetDevice(0));Logger logger;auto* builder=nvinfer1::createInferBuilder(logger);require(builder,"Builder");
    auto* network=builder->createNetworkV2(0);auto* config=builder->createBuilderConfig();require(network&&config,"Network/config");
    nvinfer1::Dims shape{};shape.nbDims=1;shape.d[0]=1;auto* input=network->addInput("input",nvinfer1::DataType::kFLOAT,shape);
    auto* identity=network->addIdentity(*input);identity->getOutput(0)->setName("output");network->markOutput(*identity->getOutput(0));
    auto* plan=builder->buildSerializedNetwork(*network,*config);require(plan,"Engine plan");auto* runtime=nvinfer1::createInferRuntime(logger);
    auto* engine=runtime->deserializeCudaEngine(plan->data(),plan->size());require(engine,"Engine");
    delete plan;delete config;delete network;delete builder;
    cudaStream_t a,b;check(cudaStreamCreateWithFlags(&a,cudaStreamNonBlocking));check(cudaStreamCreateWithFlags(&b,cudaStreamNonBlocking));
    int *counter,*other;check(cudaMalloc(&counter,4));check(cudaMalloc(&other,4));check(cudaMemset(counter,0,4));check(cudaMemset(other,0,4));
    std::vector<void*> bindings{counter},changed{other};
    {
        Garnet::TRTContextPool pool(engine,1);std::atomic<bool> gate{false};Garnet::TRTContextPool::Slot* first;
        {auto lease=pool.Acquire(a);first=lease.operator->();first->warmed=true;first->bindings=bindings;check(cudaLaunchHostFunc(a,hostGate,&gate));}
        std::atomic<bool> observed{false};
        std::thread opener([&]{auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(10);
            while(pool.GraphCounts()[5]==0&&std::chrono::steady_clock::now()<deadline)std::this_thread::yield();
            observed=pool.GraphCounts()[5]>0;gate.store(true,std::memory_order_release);});
        bool same=false;{auto lease=pool.Acquire(a,&bindings);same=lease.operator->()==first&&pool.Size()==1;}
        opener.join();check(cudaStreamSynchronize(a));require(same&&observed,"Warmup wait branch/one context missing");
        std::cout<<"SERIAL_GRAPH_POOL_WARMUP_PASS"<<std::endl;
    }
    {
        Garnet::TRTContextPool pool(engine,2);Garnet::TRTContextPool::Slot* first;
        // Create both contexts before a host gate. Context allocation may
        // synchronize the device and must never occur behind a closed gate.
        {auto lease=pool.Acquire(a);first=lease.operator->();first->warmed=true;first->bindings=bindings;capture(first,a,counter);auto second=pool.Acquire(b);}
        check(cudaStreamSynchronize(a));check(cudaStreamSynchronize(b));
        for(int test=0;test<4;++test){
            std::atomic<bool> gate{false};
            {auto lease=pool.Acquire(a,&bindings);require(lease.operator->()==first,"Expected captured first slot");
                if(test==2)first->shapeSignature={1};
                if(test==3){auto opaque=std::make_shared<int>(1);first->operatorOwner=std::shared_ptr<Garnet::OperatorExecutionOwner>(opaque,reinterpret_cast<Garnet::OperatorExecutionOwner*>(opaque.get()));}
                check(cudaLaunchHostFunc(a,hostGate,&gate));check(cudaGraphLaunch(first->executable,a));}
            bool valid=true;
            if(test==0){auto lease=pool.Acquire(a,&bindings);valid=lease.operator->()==first&&pool.GraphCounts()[4]>0;check(cudaGraphLaunch(first->executable,a));}
            {auto lease=pool.Acquire(test==1?b:a,test==0?&changed:&bindings);valid=valid&&lease.operator->()!=first;}
            gate.store(true,std::memory_order_release);check(cudaStreamSynchronize(a));check(cudaStreamSynchronize(b));
            require(valid,"Pending stream/binding/shape/provider constraint failed");first->shapeSignature.clear();first->operatorOwner.reset();
        }
        int result=0;check(cudaMemcpy(&result,counter,4,cudaMemcpyDeviceToHost));require(result==5,"Serialized graph result differs");
        std::cout<<"SERIAL_GRAPH_POOL_STREAM_BINDING_SHAPE_PROVIDER_PASS"<<std::endl;
    }
    {
        Garnet::TRTContextPool pool(engine,2);std::promise<Garnet::TRTContextPool::Slot*> ready;std::promise<void> begin;std::promise<bool> armed;
        auto go=begin.get_future();auto slot=ready.get_future();auto pending=armed.get_future();std::atomic<bool> gate{false};
        std::exception_ptr workerError;
        std::thread worker([&]{try{
            check(cudaSetDevice(0));Garnet::TRTContextPool::Slot* first;
            {auto lease=pool.Acquire(cudaStreamPerThread);first=lease.operator->();first->warmed=true;first->bindings=bindings;capture(first,cudaStreamPerThread,counter);
                ready.set_value(first);go.wait();check(cudaLaunchHostFunc(cudaStreamPerThread,hostGate,&gate));check(cudaGraphLaunch(first->executable,cudaStreamPerThread));}
            armed.set_value(true);check(cudaStreamSynchronize(cudaStreamPerThread));
        }catch(...){workerError=std::current_exception();try{ready.set_value(nullptr);}catch(...){}try{armed.set_value(false);}catch(...){}}});
        auto* first=slot.get();bool distinct=false;
        if(first){{auto lease=pool.Acquire(cudaStreamPerThread);distinct=lease.operator->()!=first;}check(cudaStreamSynchronize(cudaStreamPerThread));begin.set_value();if(pending.get()){auto lease=pool.Acquire(cudaStreamPerThread,&bindings);distinct=distinct&&lease.operator->()!=first;}}
        gate.store(true,std::memory_order_release);worker.join();if(workerError)std::rethrow_exception(workerError);
        require(distinct,"Per-thread sentinel aliased across CPU threads");
        std::cout<<"SERIAL_GRAPH_POOL_PER_THREAD_SENTINEL_PASS"<<std::endl;
    }
    check(cudaStreamSynchronize(b));check(cudaFree(counter));check(cudaFree(other));check(cudaStreamDestroy(a));check(cudaStreamDestroy(b));delete engine;delete runtime;
    std::cout<<"QWEN35_SERIAL_GRAPH_POOL_COMPLETE"<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
