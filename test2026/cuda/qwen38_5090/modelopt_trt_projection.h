// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <NvInfer.h>
namespace Garnet {
// Numerical Q/DQ reference only: tested TensorRT10.16 SM120 engines expand
// these weights and use BF16 GEMM. Do not use this for 27B deployment or
// claim native FP4 tensor-core execution from numerical agreement alone.
struct ModelOptTRTProjection {
    nvinfer1::ITensor* weight=nullptr;
    nvinfer1::ITensor* inputScale=nullptr;
    bool nvfp4=false;
};
// Constants retain caller-owned checkpoint storage until engine serialization.
// NVFP4 uses E2M1 weights, E4M3 block16 scales and F32 double scale.
ModelOptTRTProjection AddModelOptTRTWeight(nvinfer1::INetworkDefinition& network,
    nvinfer1::Dims logicalShape,nvinfer1::Weights packed,nvinfer1::Weights scales,
    nvinfer1::Weights scale2,nvinfer1::Weights inputScale,bool nvfp4);
nvinfer1::ITensor* ProjectModelOptTRT(nvinfer1::INetworkDefinition& network,
    nvinfer1::ITensor& input,const ModelOptTRTProjection& projection);
}
