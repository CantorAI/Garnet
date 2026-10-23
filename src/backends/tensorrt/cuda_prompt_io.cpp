// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_tensor_io.h"
#include "tensor_helper.h"
#include <algorithm>
#include <vector>
namespace Garnet { namespace {
void promptCheck(bool okay,const char* message){if(!okay)throw X::Error(message);}
void promptCUDA(cudaError_t status){if(status!=cudaSuccess)throw X::Error(cudaGetErrorString(status));}
uint64_t promptSpan(const X3TensorInfo& info,uint32_t rank,int device){
    promptCheck(!info.symbolic&&!info.readonly&&info.device_type==TensorHelper::CudaDevice&&
        info.device_id==device&&info.dtype==X3_TENSOR_FLOAT32&&info.data&&info.rank==rank,
        "Prompt packing requires writable contiguous FP32 tensors on the execution CUDA device");
    uint64_t bytes=4;
    for(uint32_t i=rank;i-->0;){
        promptCheck(info.shape[i]>0&&info.shape[i]<=16384,"Prompt tensor dimension outside admitted range");
        promptCheck(!info.strides||info.strides[i]==int64_t(bytes),"Prompt tensor must be contiguous");
        bytes*=uint64_t(info.shape[i]); // rank<=3, bounded dimensions.
    }
    promptCheck(info.byte_size>=bytes,"Prompt source logical footprint exceeds storage");
    return bytes;
}
}
X::Value CUDATensorIO::PackPrompt(X::Value embeddingRows,X::Value visualRows,X::Value visualStarts){
    promptCheck(embeddingRows.IsList()&&visualRows.IsList()&&visualStarts.IsList(),"Prompt packing requires three lists");
    const uint64_t batch=embeddingRows.Size();
    promptCheck((batch==1||batch==2||batch==4||batch==8)&&visualRows.Size()==batch&&visualStarts.Size()==batch,
        "Prompt packing requires matching lists and batch1/2/4/8");
    int device=0;promptCUDA(cudaGetDevice(&device));
    std::vector<X::Tensor> embeddings,visuals;
    std::vector<int64_t> lengths,starts,counts;
    std::vector<std::pair<X::Tensor,X3TensorAccess>> uses;
    int64_t width=0,tokens=0;
    // Validate every row before allocating output or enqueueing a copy.
    for(uint64_t row=0;row<batch;++row){
        auto value=embeddingRows.Get(row);promptCheck(value.runtime()==Host()->runtime,"Prompt source runtime differs");
        X::Tensor embedding(value);auto e=embedding.Info();promptSpan(e,2,device);
        promptCheck(e.shape[0]<=8192,"Prompt source exceeds8192tokens");
        if(!row)width=e.shape[1];
        promptCheck(e.shape[1]==width,"Prompt embedding widths differ");
        const auto startValue=visualStarts.Get(row);
        promptCheck(startValue.IsInt64(),"Prompt visual start must be an integer");
        const int64_t start=startValue.ToInt64();auto visualValue=visualRows.Get(row);
        X::Tensor visual;int64_t count=0;
        if(visualValue.IsNone())promptCheck(start==-1,"Missing prompt visual requires start=-1");
        else{
            promptCheck(visualValue.runtime()==Host()->runtime,"Prompt visual runtime differs");
            visual=X::Tensor(visualValue);auto v=visual.Info();promptSpan(v,3,device);
            promptCheck(v.shape[0]==1&&v.shape[2]==width&&start>=0&&start<=e.shape[0]&&v.shape[1]<=e.shape[0]-start,
                "Prompt visual shape or interval invalid");
            count=v.shape[1];uses.push_back({visual,X3_TENSOR_READ});
        }
        embeddings.push_back(embedding);visuals.push_back(visual);lengths.push_back(e.shape[0]);
        starts.push_back(start);counts.push_back(count);tokens=std::max(tokens,e.shape[0]);
        uses.push_back({embedding,X3_TENSOR_READ});
    }
    const uint64_t bytes=uint64_t(tokens)*batch*uint64_t(width)*4;
    promptCheck(bytes<=268435456ULL,"Prompt packed output exceeds256MiB");
    auto output=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{tokens,int64_t(batch),width});
    uses.push_back({output,X3_TENSOR_WRITE});
    auto use=TensorHelper::AcquireGPU(uses);auto* destination=static_cast<float*>(output.Info().data);
    const size_t rowBytes=size_t(width)*4,pitch=size_t(batch)*rowBytes;
    for(uint64_t row=0;row<batch;++row){
        promptCUDA(cudaMemcpy2DAsync(destination+row*width,pitch,embeddings[row].Info().data,rowBytes,
            rowBytes,size_t(lengths[row]),cudaMemcpyDeviceToDevice,CurrentExecutionStream()));
        if(counts[row])promptCUDA(cudaMemcpy2DAsync(destination+(uint64_t(starts[row])*batch+row)*width,pitch,
            visuals[row].Info().data,rowBytes,rowBytes,size_t(counts[row]),cudaMemcpyDeviceToDevice,CurrentExecutionStream()));
    }
    use.Finish();auto result=X::Value::List(Host());
    for(int64_t t=0;t<tokens;++t)
        result.Append(output.View({int64_t(batch),width},{int64_t(rowBytes),4},uint64_t(t)*pitch));
    return result;
}
}
#endif
