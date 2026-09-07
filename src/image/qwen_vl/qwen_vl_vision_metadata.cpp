#include "qwen_vl_vision_metadata.h"
#include "../../tensor/garnet_tensor.h"
#include "../../tensor/tensor_helper.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
#include <limits>

extern "C" cudaError_t runQwenVLVisionMetadata(
    long long* bilinearIndices,
    __nv_bfloat16* bilinearWeights,
    long long* positionIds,
    int* cuSeqlens,
    int gridT,
    int gridH,
    int gridW,
    int spatialMergeSize,
    int positionGridSide,
    cudaStream_t stream);

namespace Garnet::Image::QwenVL
{
    namespace
    {
        X::Tensor MakeDeviceTensor(
            X3PackageHost* host,
            X3TensorDType dataType,
            const std::initializer_list<int>& dimensions,
            size_t bytes,
            void** devicePointer)
        {
            auto tensor = TensorHelper::CreateGPU(host, dataType,
                std::vector<int64_t>(dimensions.begin(), dimensions.end()));
            auto info = tensor.Info();
            if (info.byte_size != bytes) throw std::logic_error("vision tensor size mismatch");
            *devicePointer = info.data;
            return tensor;
        }
    }

    VisionMetadataTensors BuildVisionMetadataTensors(
        X3PackageHost* host,
        int gridT,
        int gridH,
        int gridW,
        int spatialMergeSize,
        int numPositionEmbeddings)
    {
        if (gridT <= 0 || gridH <= 0 || gridW <= 0 || spatialMergeSize <= 0 ||
            gridH % spatialMergeSize != 0 || gridW % spatialMergeSize != 0) {
            throw std::invalid_argument("invalid Qwen vision metadata grid");
        }
        if (numPositionEmbeddings <= 0) throw std::invalid_argument("invalid position embedding count");
        const int positionGridSide = static_cast<int>(std::sqrt(numPositionEmbeddings));
        if (positionGridSide * positionGridSide != numPositionEmbeddings) {
            throw std::invalid_argument("Qwen position embedding count must be a square");
        }
        int64_t patchCount64 = 1;
        for (int dimension : {gridT, gridH, gridW}) {
            if (patchCount64 > (std::numeric_limits<int>::max)() / 4 / dimension)
                throw std::overflow_error("Qwen vision grid is too large");
            patchCount64 *= dimension;
        }
        const int patchCount = static_cast<int>(patchCount64);
        void* indicesDevice = nullptr;
        void* weightsDevice = nullptr;
        void* positionsDevice = nullptr;
        void* cuSeqlensDevice = nullptr;
        X::Tensor indices = MakeDeviceTensor(
            host, X3_TENSOR_INT64,
            {patchCount, 4},
            static_cast<size_t>(4) * patchCount * sizeof(long long),
            &indicesDevice);
        X::Tensor weights = MakeDeviceTensor(
            host, X3_TENSOR_BFLOAT16,
            {patchCount, 4},
            static_cast<size_t>(4) * patchCount * sizeof(__nv_bfloat16),
            &weightsDevice);
        X::Tensor positions = MakeDeviceTensor(
            host, X3_TENSOR_INT64,
            {patchCount, 2},
            static_cast<size_t>(patchCount) * 2 * sizeof(long long),
            &positionsDevice);
        X::Tensor cuSeqlens = MakeDeviceTensor(
            host, X3_TENSOR_INT32,
            {gridT + 1},
            static_cast<size_t>(gridT + 1) * sizeof(int),
            &cuSeqlensDevice);

        auto use = TensorHelper::AcquireGPU({
            {indices, X3_TENSOR_WRITE}, {weights, X3_TENSOR_WRITE},
            {positions, X3_TENSOR_WRITE}, {cuSeqlens, X3_TENSOR_WRITE}}, cudaStreamPerThread);
        const cudaError_t status = runQwenVLVisionMetadata(
            static_cast<long long*>(indicesDevice),
            static_cast<__nv_bfloat16*>(weightsDevice),
            static_cast<long long*>(positionsDevice),
            static_cast<int*>(cuSeqlensDevice),
            gridT,
            gridH,
            gridW,
            spatialMergeSize,
            positionGridSide,
            cudaStreamPerThread);
        use.Finish();
        if (status != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(status));
        }

        VisionMetadataTensors result;
        result.bilinearIndices = X::Value(indices);
        result.bilinearWeights = X::Value(weights);
        result.positionIds = X::Value(positions);
        result.cuSeqlens = X::Value(cuSeqlens);
        return result;
    }
}
