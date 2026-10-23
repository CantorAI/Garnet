// SPDX-License-Identifier: Apache-2.0
#include "modelopt_cublaslt.h"
#include <cuda_runtime_api.h>
#include <cmath>
namespace Garnet {
namespace {
struct Descriptors {
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulPreference_t preference=nullptr;
    ~Descriptors(){
        if(preference)cublasLtMatmulPreferenceDestroy(preference);
        if(c)cublasLtMatrixLayoutDestroy(c);
        if(b)cublasLtMatrixLayoutDestroy(b);
        if(a)cublasLtMatrixLayoutDestroy(a);
        if(operation)cublasLtMatmulDescDestroy(operation);
    }
};
}
#define MODELOPT_LT_CHECK(call) do { const auto status=(call); if(status!=CUBLAS_STATUS_SUCCESS)return status; } while(false)
cublasStatus_t RunModelOptNVFP4Matmul(cublasLtHandle_t handle,
    const void* weights,const void* weightScales,const void* input,const void* inputScales,
    void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept {
#if CUDART_VERSION >= 12080
    if(!handle||!weights||!weightScales||!input||!inputScales||!output||batch<=0||rows<=0||
        columns<=0||rows%16||columns%64||!std::isfinite(globalScale)||globalScale<=0||
        (workspaceBytes&&!workspace))return CUBLAS_STATUS_INVALID_VALUE;
    Descriptors d;
    MODELOPT_LT_CHECK(cublasLtMatmulDescCreate(&d.operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    const cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
    MODELOPT_LT_CHECK(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA)));
    MODELOPT_LT_CHECK(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB)));
    const cublasLtMatmulMatrixScale_t mode=CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    MODELOPT_LT_CHECK(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_A_SCALE_MODE,&mode,sizeof(mode)));
    MODELOPT_LT_CHECK(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_B_SCALE_MODE,&mode,sizeof(mode)));
    MODELOPT_LT_CHECK(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,&weightScales,sizeof(weightScales)));
    MODELOPT_LT_CHECK(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,&inputScales,sizeof(inputScales)));
    // Interpret W and X as column-major Kxrows and Kxbatch matrices.
    // The column-major rowsxbatch result is the desired row-major batchxrows.
    MODELOPT_LT_CHECK(cublasLtMatrixLayoutCreate(&d.a,CUDA_R_4F_E2M1,columns,rows,columns));
    MODELOPT_LT_CHECK(cublasLtMatrixLayoutCreate(&d.b,CUDA_R_4F_E2M1,columns,batch,columns));
    MODELOPT_LT_CHECK(cublasLtMatrixLayoutCreate(&d.c,CUDA_R_16BF,rows,batch,rows));
    MODELOPT_LT_CHECK(cublasLtMatmulPreferenceCreate(&d.preference));
    MODELOPT_LT_CHECK(cublasLtMatmulPreferenceSetAttribute(d.preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&workspaceBytes,sizeof(workspaceBytes)));
    cublasLtMatmulHeuristicResult_t candidates[8]{};int count=0;
    MODELOPT_LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle,d.operation,d.a,d.b,d.c,d.c,d.preference,8,candidates,&count));
    if(!count)return CUBLAS_STATUS_NOT_SUPPORTED;
    const float beta=0;
    for(int i=0;i<count;++i){
        if(candidates[i].state!=CUBLAS_STATUS_SUCCESS)continue;
        const auto status=cublasLtMatmul(handle,d.operation,&globalScale,weights,d.a,input,d.b,&beta,output,d.c,
            output,d.c,&candidates[i].algo,workspace,workspaceBytes,stream);
        if(status==CUBLAS_STATUS_SUCCESS)return status;
        if(status!=CUBLAS_STATUS_NOT_SUPPORTED)return status;
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
#else
    return CUBLAS_STATUS_NOT_SUPPORTED;
#endif
}
#undef MODELOPT_LT_CHECK
}
