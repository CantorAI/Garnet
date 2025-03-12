#include <cuda_fp16.h>
#include <cuda_bf16.h>

// Templated kernel for element-wise type conversion.
template <typename InType, typename OutType>
__global__ void astypeKernel(const InType* __restrict__ input,
    OutType* __restrict__ output,
    int num_elements)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = static_cast<OutType>(input[idx]);
    }
}

// Extern "C" interface for converting from __nv_bfloat16 to float.
extern "C" {

    // Launch function for converting __nv_bfloat16 to float.
    void runAstype_bf16_to_fp32(const __nv_bfloat16* input, float* output, int num_elements)
    {
        int blockSize = 256;
        int gridSize = (num_elements + blockSize - 1) / blockSize;
        astypeKernel<__nv_bfloat16, float> << <gridSize, blockSize >> > (input, output, num_elements);
        cudaDeviceSynchronize();
    }

    // You can add additional conversion functions if needed.
    // For example, converting from __half to float:
    void runAstype_fp16_to_fp32(const __half* input, float* output, int num_elements)
    {
        int blockSize = 256;
        int gridSize = (num_elements + blockSize - 1) / blockSize;
        astypeKernel<__half, float> << <gridSize, blockSize >> > (input, output, num_elements);
        cudaDeviceSynchronize();
    }
}
