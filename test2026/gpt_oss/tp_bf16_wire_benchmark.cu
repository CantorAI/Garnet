// SPDX-License-Identifier: Apache-2.0
// Native transport screen only; no TensorRT/XModel dispatch changes.
#include "gpt_oss_kernels.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

#define CUDA_OK(call) do { const auto status=(call); if(status!=cudaSuccess) { \
    std::fprintf(stderr,"%s: %s\n",#call,cudaGetErrorString(status)); std::exit(1); } } while(0)

namespace {
uint32_t bits(float value) {uint32_t out;std::memcpy(&out,&value,4);return out;}
float fromBits(uint32_t value) {float out;std::memcpy(&out,&value,4);return out;}
// Independent host integer round-to-nearest/ties-to-even, not CUDA conversion.
float roundBf16(float value) {
    uint32_t raw=bits(value);
    if((raw&0x7fffffffU)>0x7f800000U)return fromBits((raw&0xffff0000U)|0x00400000U);
    return fromBits((raw+0x7fffU+((raw>>16)&1U))&0xffff0000U);
}
float sumOracle(float a,float b) {volatile float sum=a+b;return roundBf16(sum);}
uint16_t finiteBits(uint16_t value) {
    return (value&0x7f80U)==0x7f80U?uint16_t(value^0x0080U):value;
}
uint16_t boundaryBits(int i,int rank) {
    const uint16_t base=finiteBits(uint16_t(i));
    if(!rank)return base;
    switch((i/65536)%6) {
        case 0:return base;
        case 1:return uint16_t(base^0x8000U);
        case 2:return finiteBits(uint16_t(base+1));
        case 3:return uint16_t(base&0x8000U);
        case 4:return uint16_t((base&0x8000U)|1U);
        default:return uint16_t((base&0x8000U)|0x3f80U);
    }
}
__global__ void changeInput(float* input,int count,float increment) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x)
        input[i]=__bfloat162float(__float2bfloat16_rn(input[i]+increment));
}
int screen(int batch,int repeats,bool bf16) {
    constexpr int operations=72;
    const int count=batch*2880;
    CUDA_OK(Garnet::GptOssTpAcquire());
    std::array<cudaStream_t,2> streams{};
    std::array<float*,2> inputs{},outputs{};
    std::array<void*,2> scratch{};
    std::array<cudaGraph_t,2> graphs{};
    std::array<cudaGraphExec_t,2> executions{};
    std::array<std::vector<float>,2> host{std::vector<float>(count),std::vector<float>(count)};
    auto paired=[&](auto work) {
        std::array<std::thread,2> workers;
        for(int rank=0;rank<2;++rank)workers[rank]=std::thread([&,rank] {
            CUDA_OK(cudaSetDevice(rank));work(rank);
        });
        for(auto& worker:workers)worker.join();
    };
    auto reduce=[&](int rank) {
        if(bf16)CUDA_OK(Garnet::GptOssTpAllReduceBf16(inputs[rank],outputs[rank],
            scratch[rank],count,rank,streams[rank]));
        else CUDA_OK(Garnet::GptOssTpAllReduce(inputs[rank],outputs[rank],count,rank,streams[rank]));
    };
    for(int rank=0;rank<2;++rank) {
        CUDA_OK(cudaSetDevice(rank));
        CUDA_OK(cudaStreamCreateWithFlags(&streams[rank],cudaStreamNonBlocking));
        CUDA_OK(cudaMalloc(&inputs[rank],count*sizeof(float)));
        CUDA_OK(cudaMalloc(&outputs[rank],count*sizeof(float)));
        CUDA_OK(cudaMalloc(&scratch[rank],2*count*sizeof(uint16_t)));
        // API rejection must precede any memory access/collective submission.
        for(int badRank:{-1,2})
            if(Garnet::GptOssTpAllReduceBf16(inputs[rank],outputs[rank],scratch[rank],count,
                badRank,streams[rank])!=cudaErrorInvalidValue)return 4;
        if(Garnet::GptOssTpAllReduceBf16(inputs[rank],outputs[rank],scratch[rank],count,
            1-rank,streams[rank])!=cudaErrorInvalidDevice)return 4;
        for(size_t rejected:{size_t(0),SIZE_MAX/4+1})
            if(Garnet::GptOssTpAllReduceBf16(inputs[rank],outputs[rank],scratch[rank],rejected,
                rank,streams[rank])!=cudaErrorInvalidValue)return 4;
        if(Garnet::GptOssTpAllReduceBf16(nullptr,outputs[rank],scratch[rank],count,rank,streams[rank])!=cudaErrorInvalidValue ||
           Garnet::GptOssTpAllReduceBf16(inputs[rank],nullptr,scratch[rank],count,rank,streams[rank])!=cudaErrorInvalidValue ||
           Garnet::GptOssTpAllReduceBf16(inputs[rank],outputs[rank],nullptr,count,rank,streams[rank])!=cudaErrorInvalidValue)return 4;
    }
    auto upload=[&] {
        for(int rank=0;rank<2;++rank) {
            CUDA_OK(cudaSetDevice(rank));
            CUDA_OK(cudaMemcpyAsync(inputs[rank],host[rank].data(),count*sizeof(float),
                cudaMemcpyHostToDevice,streams[rank]));
            CUDA_OK(cudaStreamSynchronize(streams[rank]));
        }
    };
    auto check=[&](const char* label) {
        std::vector<float> result(count);
        for(int rank=0;rank<2;++rank) {
            CUDA_OK(cudaSetDevice(rank));
            CUDA_OK(cudaMemcpy(result.data(),outputs[rank],count*sizeof(float),cudaMemcpyDeviceToHost));
            for(int i=0;i<count;++i) {
                const float expected=sumOracle(host[0][i],host[1][i]);
                const float actual=bf16?result[i]:roundBf16(result[i]);
                if(bits(actual)!=bits(expected)) {
                    std::fprintf(stderr,"%s rank=%d index=%d input=%08x/%08x got=%08x expected=%08x\n",
                        label,rank,i,bits(host[0][i]),bits(host[1][i]),bits(actual),bits(expected));
                    std::exit(3);
                }
            }
        }
    };
    // Six pair families, each sweeps all finite BF16 sign/exponent/mantissa bits.
    for(int family=0;family<6;++family) {
        for(int rank=0;rank<2;++rank)for(int i=0;i<count;++i)
            host[rank][i]=fromBits(uint32_t(boundaryBits((i%65536)+family*65536,rank))<<16);
        upload();paired([&](int rank){reduce(rank);CUDA_OK(cudaStreamSynchronize(streams[rank]));});
        check("finite-boundary");
    }
    std::puts("Both-rank finite BF16 bit sweep, cancellation/rounding/overflow/zero and API rejection passed");
    auto reset=[&](int trial) {
        for(int rank=0;rank<2;++rank)for(int i=0;i<count;++i)
            host[rank][i]=roundBf16(float((i*(rank?31:17)+trial)%251-125)/128.f);
        upload();
    };
    reset(0);
    paired([&](int rank){for(int i=0;i<10;++i)reduce(rank);CUDA_OK(cudaStreamSynchronize(streams[rank]));});
    check("eager");
    auto capture=[&](bool changing) {
        paired([&](int rank) {
            CUDA_OK(cudaStreamBeginCapture(streams[rank],cudaStreamCaptureModeThreadLocal));
            for(int i=0;i<operations;++i) {
                if(changing) {
                    changeInput<<<32,256,0,streams[rank]>>>(inputs[rank],count,rank?-.0078125f:.015625f);
                    CUDA_OK(cudaGetLastError());
                }
                reduce(rank);
            }
            CUDA_OK(cudaStreamEndCapture(streams[rank],&graphs[rank]));
            CUDA_OK(cudaGraphInstantiate(&executions[rank],graphs[rank],0));
        });
    };
    auto replay=[&](int times) {
        const auto start=std::chrono::steady_clock::now();
        paired([&](int rank) {
            for(int i=0;i<times;++i)CUDA_OK(cudaGraphLaunch(executions[rank],streams[rank]));
            CUDA_OK(cudaStreamSynchronize(streams[rank]));
        });
        return std::chrono::duration<double,std::micro>(std::chrono::steady_clock::now()-start).count()/(times*operations);
    };
    auto releaseGraph=[&] {
        for(int rank=0;rank<2;++rank) {
            CUDA_OK(cudaSetDevice(rank));CUDA_OK(cudaGraphExecDestroy(executions[rank]));CUDA_OK(cudaGraphDestroy(graphs[rank]));
        }
    };
    capture(true);
    for(int trial=0;trial<5;++trial) {
        reset(trial);replay(3);
        for(int step=0;step<3*operations;++step)for(int rank=0;rank<2;++rank)
            for(float& value:host[rank])value=roundBf16(value+(rank?-.0078125f:.015625f));
        check("changing-graph");
    }
    releaseGraph();capture(false);replay(10);
    std::array<double,5> timings{};
    for(int trial=0;trial<5;++trial) {reset(trial);timings[trial]=replay(repeats);check("timed-graph");}
    auto sorted=timings;std::sort(sorted.begin(),sorted.end());
    std::printf("batch=%d mode=%s parity=PASS graph_ops=%d us_per_op=",batch,bf16?"bf16-nccl":"fp32-nccl",operations);
    for(double value:timings)std::printf(" %.3f",value);
    std::printf(" median=%.3f\n",sorted[2]);
    releaseGraph();
    for(int rank=0;rank<2;++rank) {
        CUDA_OK(cudaSetDevice(rank));CUDA_OK(cudaFree(inputs[rank]));CUDA_OK(cudaFree(outputs[rank]));
        CUDA_OK(cudaFree(scratch[rank]));CUDA_OK(cudaStreamDestroy(streams[rank]));
    }
    Garnet::GptOssTpRelease();return 0;
}
}
int main(int argc,char** argv) {
    if(argc!=4)return 2;
    const int batch=std::atoi(argv[1]),repeats=std::atoi(argv[3]);
    if(batch<128||batch>512||repeats<1||repeats>1000)return 2;
    const bool bf16=std::strcmp(argv[2],"bf16")==0;
    if(!bf16&&std::strcmp(argv[2],"fp32")!=0)return 2;
    // This screen must use NCCL, never a direct-transport fallback.
    if(const char* direct=std::getenv("GARNET_GPT_OSS_DIRECT_ALLREDUCE"))
        if(std::strcmp(direct,"0")!=0)return 2;
    for(int cycle=0;cycle<2;++cycle) {const int status=screen(batch,repeats,bf16);if(status)return status;}
    return 0;
}
