// SPDX-License-Identifier: Apache-2.0
#include "modelopt_cublaslt.h"
#include <cuda_runtime_api.h>
#include <cmath>
#include <mutex>
#include <unordered_map>
#include <cstdint>
namespace Garnet {
namespace {
struct Context {
    int device=-1;cublasLtHandle_t handle=nullptr;std::mutex mutex;
    ~Context(){
        int previous=-1;cudaGetDevice(&previous);cudaSetDevice(device);
        if(handle)cublasLtDestroy(handle);if(previous>=0)cudaSetDevice(previous);
    }
};
std::mutex contextsMutex;
std::unordered_map<int,std::weak_ptr<Context>> contexts;
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
#if CUDART_VERSION >= 12080
template<class T> bool ConfigEquals(const cublasLtMatmulAlgo_t& algo,cublasLtMatmulAlgoConfigAttributes_t attribute,T expected){
    T value{};size_t written=0;
    return cublasLtMatmulAlgoConfigGetAttribute(&algo,attribute,&value,sizeof(value),&written)==CUBLAS_STATUS_SUCCESS&&written==sizeof(value)&&value==expected;
}
bool IsDownCandidate(const cublasLtMatmulAlgo_t& algo){
    return ConfigEquals<int32_t>(algo,CUBLASLT_ALGO_CONFIG_ID,70)&&ConfigEquals<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_TILE_ID,20)&&
        ConfigEquals<int32_t>(algo,CUBLASLT_ALGO_CONFIG_SPLITK_NUM,-2)&&ConfigEquals<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,2)&&
        ConfigEquals<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,0)&&ConfigEquals<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,0)&&
        ConfigEquals<uint32_t>(algo,CUBLASLT_ALGO_CONFIG_STAGES_ID,37)&&ConfigEquals<uint16_t>(algo,CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,0)&&
        ConfigEquals<uint16_t>(algo,CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,5);
}
#endif
}
std::shared_ptr<void> CreateModelOptNVFP4Context(){
#if CUDART_VERSION >= 12080
    int device=-1;if(cudaGetDevice(&device)!=cudaSuccess)return {};
    std::lock_guard<std::mutex> lock(contextsMutex);
    if(auto owner=contexts[device].lock())return owner;
    cudaDeviceProp properties{};
    if(cudaGetDeviceProperties(&properties,device)!=cudaSuccess||properties.major<10)return {};
    auto owner=std::make_shared<Context>();owner->device=device;
    if(cublasLtCreate(&owner->handle)!=CUBLAS_STATUS_SUCCESS)return {};
    contexts[device]=owner;return owner;
#else
    return {};
#endif
}
cublasStatus_t RunModelOptNVFP4MatmulContext(const std::shared_ptr<void>& opaque,
    const void* weights,const void* weightScales,const void* input,const void* inputScales,
    void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream,bool downCandidate) noexcept {
    if(!opaque)return CUBLAS_STATUS_NOT_INITIALIZED;
    try{
        auto context=std::static_pointer_cast<Context>(opaque);
        std::lock_guard<std::mutex> lock(context->mutex);
        return RunModelOptNVFP4Matmul(context->handle,weights,weightScales,input,inputScales,
            output,batch,rows,columns,globalScale,workspace,workspaceBytes,stream,downCandidate);
    }catch(...){return CUBLAS_STATUS_INTERNAL_ERROR;}
}
cublasStatus_t RunModelOptFP8MatmulContext(const std::shared_ptr<void>& opaque,
    const void* weights,const void* input,void* output,int batch,int rows,int columns,
    float globalScale,void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept {
    if(!opaque)return CUBLAS_STATUS_NOT_INITIALIZED;
    try{
        auto context=std::static_pointer_cast<Context>(opaque);
        std::lock_guard<std::mutex> lock(context->mutex);
        return RunModelOptFP8Matmul(context->handle,weights,input,output,batch,rows,columns,
            globalScale,workspace,workspaceBytes,stream);
    }catch(...){return CUBLAS_STATUS_INTERNAL_ERROR;}
}
#define MODELOPT_LT_CHECK(call) do { const auto status=(call); if(status!=CUBLAS_STATUS_SUCCESS)return status; } while(false)
cublasStatus_t RunModelOptNVFP4Matmul(cublasLtHandle_t handle,
    const void* weights,const void* weightScales,const void* input,const void* inputScales,
    void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream,bool downCandidate) noexcept {
#if CUDART_VERSION >= 12080
    if(!handle||!weights||!weightScales||!input||!inputScales||!output||batch<=0||rows<=0||
        columns<=0||rows%16||columns%64||!std::isfinite(globalScale)||globalScale<=0||
        (workspaceBytes&&!workspace))return CUBLAS_STATUS_INVALID_VALUE;
    // Explicit research policy, qualified on cuBLASLt13.4. Never silently
    // substitute another algorithm when this policy was requested.
    if(downCandidate&&(rows!=5120||columns!=17408||(batch!=1&&batch!=4&&batch!=128)||
        workspaceBytes!=(64ULL<<20)||cublasLtGetVersion()!=130400))return CUBLAS_STATUS_NOT_SUPPORTED;
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
    cublasLtMatmulHeuristicResult_t candidates[32]{};int count=0;
    MODELOPT_LT_CHECK(cublasLtMatmulAlgoGetHeuristic(handle,d.operation,d.a,d.b,d.c,d.c,d.preference,downCandidate?32:8,candidates,&count));
    if(!count)return CUBLAS_STATUS_NOT_SUPPORTED;
    const float beta=0;
    for(int i=0;i<count;++i){
        if(candidates[i].state!=CUBLAS_STATUS_SUCCESS)continue;
        if(downCandidate&&!IsDownCandidate(candidates[i].algo))continue;
        if(downCandidate&&candidates[i].workspaceSize>workspaceBytes)return CUBLAS_STATUS_NOT_SUPPORTED;
        const auto status=cublasLtMatmul(handle,d.operation,&globalScale,weights,d.a,input,d.b,&beta,output,d.c,
            output,d.c,&candidates[i].algo,workspace,workspaceBytes,stream);
        if(status==CUBLAS_STATUS_SUCCESS)return status;
        if(downCandidate)return status;
        if(status!=CUBLAS_STATUS_NOT_SUPPORTED)return status;
    }
    return CUBLAS_STATUS_NOT_SUPPORTED;
#else
    return CUBLAS_STATUS_NOT_SUPPORTED;
#endif
}
cublasStatus_t RunModelOptFP8Matmul(cublasLtHandle_t handle,
    const void* weights,const void* input,
    void* output,int batch,int rows,int columns,float globalScale,
    void* workspace,size_t workspaceBytes,cudaStream_t stream) noexcept {
#if CUDART_VERSION >= 12080
    if(!handle||!weights||!input||!output||batch<=0||rows<=0||
        columns<=0||rows%16||columns%16||!std::isfinite(globalScale)||globalScale<=0||
        (workspaceBytes&&!workspace))return CUBLAS_STATUS_INVALID_VALUE;
    Descriptors d;
    MODELOPT_LT_CHECK(cublasLtMatmulDescCreate(&d.operation,CUBLAS_COMPUTE_32F,CUDA_R_32F));
    const cublasOperation_t transA=CUBLAS_OP_T,transB=CUBLAS_OP_N;
    MODELOPT_LT_CHECK(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_TRANSA,&transA,sizeof(transA)));
    MODELOPT_LT_CHECK(cublasLtMatmulDescSetAttribute(d.operation,CUBLASLT_MATMUL_DESC_TRANSB,&transB,sizeof(transB)));
    // Tensorwide scalar scales are combined in alpha. Null scale pointers mean1.
    // Interpret W and X as column-major Kxrows and Kxbatch matrices.
    // The column-major rowsxbatch result is the desired row-major batchxrows.
    MODELOPT_LT_CHECK(cublasLtMatrixLayoutCreate(&d.a,CUDA_R_8F_E4M3,columns,rows,columns));
    MODELOPT_LT_CHECK(cublasLtMatrixLayoutCreate(&d.b,CUDA_R_8F_E4M3,columns,batch,columns));
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
