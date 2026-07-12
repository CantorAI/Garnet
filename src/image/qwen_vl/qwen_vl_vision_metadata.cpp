#include "qwen_vl_vision_metadata.h"
#include "../../tensor/garnet_tensor.h"
#include "../../tensor/tensor_helper.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

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
            X::TensorDataType dataType,
            const std::initializer_list<int>& dimensions,
            size_t bytes,
            void** devicePointer)
        {
            X::Tensor tensor(X::g_pXHost->CreateTensor());
            X::Port::vector<int> shape(static_cast<int>(dimensions.size()));
            for (const int dimension : dimensions) shape.push_back(dimension);
            tensor->SetDataType(dataType);
            tensor->SetShape(shape);
            if (cudaMalloc(devicePointer, bytes) != cudaSuccess ||
                TensorHelper::AttachGPUMemory(tensor, *devicePointer) != TensorOpStatus::Success) {
                if (*devicePointer) cudaFree(*devicePointer);
                *devicePointer = nullptr;
                throw std::runtime_error("failed to allocate Qwen vision metadata tensor");
            }
            return tensor;
        }
    }

    VisionMetadataTensors BuildVisionMetadataTensors(
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
        const int positionGridSide = static_cast<int>(std::sqrt(numPositionEmbeddings));
        if (positionGridSide * positionGridSide != numPositionEmbeddings) {
            throw std::invalid_argument("Qwen position embedding count must be a square");
        }
        const int patchCount = gridT * gridH * gridW;
        void* indicesDevice = nullptr;
        void* weightsDevice = nullptr;
        void* positionsDevice = nullptr;
        void* cuSeqlensDevice = nullptr;
        X::Tensor indices = MakeDeviceTensor(
            X::TensorDataType::LONGLONG,
            {patchCount, 4},
            static_cast<size_t>(4) * patchCount * sizeof(long long),
            &indicesDevice);
        X::Tensor weights = MakeDeviceTensor(
            X::TensorDataType::BFLOAT16,
            {patchCount, 4},
            static_cast<size_t>(4) * patchCount * sizeof(__nv_bfloat16),
            &weightsDevice);
        X::Tensor positions = MakeDeviceTensor(
            X::TensorDataType::LONGLONG,
            {patchCount, 2},
            static_cast<size_t>(patchCount) * 2 * sizeof(long long),
            &positionsDevice);
        X::Tensor cuSeqlens = MakeDeviceTensor(
            X::TensorDataType::INT,
            {gridT + 1},
            static_cast<size_t>(gridT + 1) * sizeof(int),
            &cuSeqlensDevice);

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
