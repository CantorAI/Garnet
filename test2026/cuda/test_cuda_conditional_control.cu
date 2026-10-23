// SPDX-License-Identifier: Apache-2.0
// Unintegrated GPU-control/lifetime probe. No model math or speed assertion.
#include <cuda_runtime.h>
#include <nlohmann/json.hpp>
#include <algorithm>
#include <iostream>
#include <stdexcept>
#include <vector>
using json=nlohmann::json;
namespace {
void check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct State{int limits[8],counts[8],error,iterations;};
__global__ void initialize(State* s,int batch,cudaGraphConditionalHandle handle){
    __shared__ int active,invalid;
    if(!threadIdx.x){active=invalid=0;s->iterations=0;s->error=0;}
    __syncthreads();
    if(threadIdx.x<batch){int n=s->limits[threadIdx.x];s->counts[threadIdx.x]=0;
        if(n<0||n>32)atomicOr(&invalid,1);else if(n)atomicOr(&active,1);}
    __syncthreads();
    if(!threadIdx.x){s->error=invalid;cudaGraphSetConditional(handle,invalid?0:active);}
}
__global__ void advance(State* s,int batch,cudaGraphConditionalHandle handle){
    if(s->iterations>=32){if(!threadIdx.x){s->error=1;cudaGraphSetConditional(handle,0);}return;}
    __shared__ int active;
    if(!threadIdx.x)active=0;
    __syncthreads();
    if(threadIdx.x<batch){int row=threadIdx.x;
        if(s->counts[row]<s->limits[row])++s->counts[row];
        if(s->counts[row]<s->limits[row])atomicOr(&active,1);}
    __syncthreads();
    if(!threadIdx.x){++s->iterations;cudaGraphSetConditional(handle,active);}
}
struct Graph{
    State* state=nullptr;cudaStream_t stream=nullptr;cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
    explicit Graph(int batch){
        check(cudaMalloc(&state,sizeof(State)));check(cudaStreamCreate(&stream));check(cudaGraphCreate(&graph,0));
        cudaGraphConditionalHandle handle;
        check(cudaGraphConditionalHandleCreate(&handle,graph,0,cudaGraphCondAssignDefault));
        cudaGraphNodeParams init{};init.type=cudaGraphNodeTypeKernel;
        init.kernel.func=(void*)initialize;init.kernel.gridDim=dim3(1);init.kernel.blockDim=dim3(32);
        void* args[]={&state,&batch,&handle};init.kernel.kernelParams=args;
        cudaGraphNode_t first,condition;
        check(cudaGraphAddNode(&first,graph,nullptr,nullptr,0,&init));
        cudaGraphNodeParams branch{};branch.type=cudaGraphNodeTypeConditional;
        branch.conditional.handle=handle;branch.conditional.type=cudaGraphCondTypeWhile;branch.conditional.size=1;
        check(cudaGraphAddNode(&condition,graph,&first,nullptr,1,&branch));
        cudaGraph_t captured;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeGlobal));
        advance<<<1,32,0,stream>>>(state,batch,handle);check(cudaPeekAtLastError());
        check(cudaStreamEndCapture(stream,&captured));
        cudaGraphNode_t child;
        check(cudaGraphAddChildGraphNode(&child,branch.conditional.phGraph_out[0],nullptr,0,captured));
        // Parent owns its clone; the capture's original graph is destroyed now.
        check(cudaGraphDestroy(captured));
        check(cudaGraphInstantiate(&executable,graph,0));
    }
    ~Graph(){if(executable)cudaGraphExecDestroy(executable);if(graph)cudaGraphDestroy(graph);if(stream)cudaStreamDestroy(stream);if(state)cudaFree(state);}
    State run(State input){
        check(cudaMemcpyAsync(state,&input,sizeof(State),cudaMemcpyHostToDevice,stream));
        check(cudaGraphLaunch(executable,stream));
        State result;check(cudaMemcpyAsync(&result,state,sizeof(State),cudaMemcpyDeviceToHost,stream));
        check(cudaStreamSynchronize(stream));return result;
    }
};
}
int main(){try{
    check(cudaSetDevice(0));cudaDeviceProp device{};check(cudaGetDeviceProperties(&device,0));
    json cases=json::array();
    for(int batch:{1,2,4,8}){
        Graph graph(batch);
        for(int pattern=0;pattern<7;++pattern)for(int repeat=0;repeat<3;++repeat){
            State input{};input.error=99;input.iterations=99;
            for(int row=0;row<batch;++row){
                input.counts[row]=777;
                input.limits[row]=pattern==0?0:pattern==1?1:pattern==2?7:pattern==3?32:
                    pattern==4?(row*7+repeat)%33:pattern==5?(row==0?-1:7):(row==batch-1?33:7);
            }
            auto actual=graph.run(input);bool invalid=pattern>=5;
            int expectedIterations=invalid?0:*std::max_element(input.limits,input.limits+batch);
            if(actual.error!=int(invalid)||actual.iterations!=expectedIterations)throw std::runtime_error("Conditional loop status/reference differs");
            for(int row=0;row<batch;++row)if(actual.counts[row]!=(invalid?0:input.limits[row]))throw std::runtime_error("Conditional loop row/reset differs");
            cases.push_back({{"batch",batch},{"pattern",pattern},{"repeat",repeat},{"iterations",actual.iterations},{"error",actual.error}});
        }
    }
    std::cout<<json{{"device",device.name},{"sm",device.major*10+device.minor},{"cases",cases},{"passed",true},
        {"source_capture_destroyed_before_launch",true},{"scope","Conditional WHILE with captured child: bounded counters, repeated reset, mixed rows and invalid limits only. No model math, proposal oracle or inference speed claim."}}.dump()<<"\n";
    std::cout<<"CUDA_CONDITIONAL_CONTROL_COMPLETE\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}
