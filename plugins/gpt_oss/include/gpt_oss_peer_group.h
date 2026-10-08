// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "operator_execution_bridge.h"
#include <cuda_runtime.h>
#include <cstddef>
namespace Garnet {
const GarnetOperatorExecutionServices* GptOssPeerExecutionServices();
cudaError_t GptOssCreatePeerGroup(const char* options,size_t bytes,void** owner);
cudaError_t GptOssPeerGroupAllReduce(const float*,float*,size_t,int,int,cudaStream_t);
}
