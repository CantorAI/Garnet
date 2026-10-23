// SPDX-License-Identifier: Apache-2.0
// Independent CUDA allocation/capture ordering control. No TensorRT or Garnet.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
static void check(cudaError_t status,const char* operation){
    if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
__global__ void increment(float* values,int count){
    int index=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(index<count)values[index]+=1.0f;
}
int main(int argc,char** argv){
    if(argc!=4&&argc!=5)return 2;
    const bool asynchronous=std::atoi(argv[1])!=0,perThread=std::atoi(argv[2])!=0,prewait=std::atoi(argv[3])!=0;
    const bool inside=argc==5&&std::atoi(argv[4])!=0;
    if(inside&&!asynchronous)return 2;
    try{
        check(cudaSetDevice(0),"device");cudaStream_t stream=cudaStreamPerThread;
        if(!perThread)check(cudaStreamCreateWithFlags(&stream,cudaStreamNonBlocking),"stream");
        constexpr int count=4096;float* data=nullptr;float* observed=nullptr;
        if(inside)check(cudaMalloc(&observed,count*sizeof(float)),"allocate_readback");
        else{
            if(asynchronous)check(cudaMallocAsync(&data,count*sizeof(float),stream),"allocate_async");
            else check(cudaMalloc(&data,count*sizeof(float)),"allocate_sync");
            observed=data;check(cudaMemsetAsync(data,0,count*sizeof(float),stream),"initialize");
        }
        cudaGraph_t graph=nullptr;cudaGraphExec_t executable=nullptr;
        check(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal),"capture_begin");
        if(inside){check(cudaMallocAsync(&data,count*sizeof(float),stream),"captured_allocate");check(cudaMemsetAsync(data,0,count*sizeof(float),stream),"captured_initialize");}
        increment<<<count/256,256,0,stream>>>(data,count);check(cudaGetLastError(),"capture_kernel");
        if(inside){check(cudaMemcpyAsync(observed,data,count*sizeof(float),cudaMemcpyDeviceToDevice,stream),"captured_copy");check(cudaFreeAsync(data,stream),"captured_free");}
        check(cudaStreamEndCapture(stream,&graph),"capture_end");
        std::printf("ALLOCATION async=%d per_thread=%d prewait=%d inside=%d pointer=%p bytes=%zu\n",asynchronous,perThread,prewait,inside,static_cast<void*>(data),count*sizeof(float));std::fflush(stdout);
        check(cudaGraphInstantiate(&executable,graph,0),"instantiate");
        if(prewait)check(cudaStreamSynchronize(stream),"prelaunch_wait");
        for(int repeat=0;repeat<3;++repeat)check(cudaGraphLaunch(executable,stream),"launch");
        std::vector<float> output(count);check(cudaMemcpyAsync(output.data(),observed,count*sizeof(float),cudaMemcpyDeviceToHost,stream),"readback");
        check(cudaStreamSynchronize(stream),"complete");
        for(float value:output)if(value!=(inside?1.0f:3.0f))throw std::runtime_error("wrong replay result");
        check(cudaGraphExecDestroy(executable),"destroy_exec");check(cudaGraphDestroy(graph),"destroy_graph");
        if(inside)check(cudaFree(observed),"free_readback");
        else if(asynchronous)check(cudaFreeAsync(data,stream),"free_async");else check(cudaFree(data),"free_sync");
        check(cudaStreamSynchronize(stream),"retire");if(!perThread)check(cudaStreamDestroy(stream),"destroy_stream");
        std::puts("CUDA_GRAPH_ALLOCATION_CONTROL_PASS");return 0;
    }catch(const std::exception& error){std::fprintf(stderr,"%s\n",error.what());return 1;}
}
