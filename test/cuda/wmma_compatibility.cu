// Compile once with compute_75 PTX and once with a native SM80+ target.
// Both paths must match the existing WMMA A-row / B-column tile layout.
#include "../../src/cuda/matrixMulTensorCore_kernel.cu"
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <vector>

static void check(cudaError_t error) {
    if (error != cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(error)); std::exit(1); }
}
int main() {
    int devices=0; if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices) return 77;
    for (int size : {16,32}) {
        std::vector<__nv_bfloat16> a(size*size),b(size*size);
        std::vector<float> output(size*size),reference(size*size);
        for (int i=0;i<size*size;++i) {
            a[i]=__float2bfloat16(float(i%7-3));
            b[i]=__float2bfloat16(float((i*3+1)%11-5));
        }
        for (int row=0;row<size;++row) for (int col=0;col<size;++col)
            for (int tile=0;tile<size;tile+=16) for (int k=0;k<16;++k)
                reference[row*size+col]+=__bfloat162float(a[row*size+tile+k]) *
                    __bfloat162float(b[tile*size+(col/16)*16+(col%16)*size+k]);
        __nv_bfloat16 *da,*db; float* dc;
        check(cudaMalloc(&da,a.size()*sizeof(a[0])));check(cudaMalloc(&db,b.size()*sizeof(b[0])));
        check(cudaMalloc(&dc,output.size()*sizeof(float)));
        check(cudaMemcpy(da,a.data(),a.size()*sizeof(a[0]),cudaMemcpyHostToDevice));
        check(cudaMemcpy(db,b.data(),b.size()*sizeof(b[0]),cudaMemcpyHostToDevice));
        check(cudaMemset(dc,0,output.size()*sizeof(float)));
        runWmmaGemmKernel_bf16(da,db,dc,size,size,size);
        check(cudaGetLastError());check(cudaDeviceSynchronize());
        check(cudaMemcpy(output.data(),dc,output.size()*sizeof(float),cudaMemcpyDeviceToHost));
        for (int i=0;i<size*size;++i) if (std::fabs(output[i]-reference[i])>0.001f) {
            std::fprintf(stderr,"BF16 mismatch at %d: %f vs %f\n",i,output[i],reference[i]); return 1;
        }
        check(cudaFree(da));check(cudaFree(db));check(cudaFree(dc));
        std::printf("BF16 tile compatibility %dx%d: PASS\n",size,size);
    }
}
