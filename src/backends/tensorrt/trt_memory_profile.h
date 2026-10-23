// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <mutex>
#include <string>

namespace Garnet {
// Allocation checkpoints, not isolated process peaks. Keep synchronization out
// of this diagnostic: engine/context creation already has explicit lifetimes.
inline bool TRTMemoryProfileEnabled() {
    const char* value=std::getenv("GARNET_TRT_LOG_ENGINE_MEMORY");
    return value && std::strcmp(value,"1")==0;
}
inline void LogTRTMemoryCheckpoint(const char* stage,const std::string& engine,
    size_t contextUpperBound=0,size_t contextCount=0) {
    if(!TRTMemoryProfileEnabled())return;
    int device=-1;size_t free=0,total=0;
    if(cudaGetDevice(&device)!=cudaSuccess || cudaMemGetInfo(&free,&total)!=cudaSuccess)return;
    static std::mutex outputMutex;
    std::lock_guard<std::mutex> guard(outputMutex);
    std::cout<<"[TRTMemoryCheckpoint] stage="<<stage<<" device="<<device
        <<" device_used_bytes="<<(total-free)<<" device_total_bytes="<<total
        <<" context_device_memory_upper_bound_bytes="<<contextUpperBound
        <<" context_count="<<contextCount<<" engine="<<engine<<std::endl;
}
}
