// SPDX-License-Identifier: Apache-2.0
#if !defined(GARNET_CPU_ONLY) && !defined(GARNET_METAL)
#include "cuda_tensor_io.h"
#include "bf16_projection_plugin.h"
#include "modelopt_cublaslt.h"
#include "tensor_helper.h"
#include "cuda_lib.h"
#include "../../cuda/qwen35_indexed_head.cuh"
#include "../../cuda/qwen35_prefix_commit.cuh"
#include "../../cuda/modelopt_nvfp4.cuh"
#include <nlohmann/json.hpp>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <vector>
extern "C" int GarnetQwenVLPreprocessJpegFileDevice(const char*,int,int,void**,size_t*,long long*,int*,int*,int*,int*,int*,int*,char*,int);
namespace Garnet {
cudaError_t CopyQwen35ActiveRows(const float*,const float*,float*,int,int,cudaStream_t);
cudaError_t CopyQwen35LastChunk(const float*,const float*,float*,int,int,int,cudaStream_t);
cudaError_t WriteQwen35Request(float*,float*,float*,const float*,const float*,const float*,int,cudaStream_t);
cudaError_t GatherQwen35Chunk(const void*,float*,const int*,int,int,int,int,cudaStream_t);
namespace {
void Check(cudaError_t status){if(status!=cudaSuccess)throw X::Error(cudaGetErrorString(status));}
std::vector<int64_t> Shape(X::Value value){
    if(!value.IsList()||!value.Size())throw X::Error("CUDA shape must be a nonempty list");
    std::vector<int64_t> result;size_t count=1;
    for(uint64_t i=0;i<value.Size();++i){auto d=value.Get(i).ToInt64();
        if(d<=0||d>INT_MAX||count>INT_MAX/size_t(d))throw X::Error("CUDA shape exceeds admitted size");
        count*=size_t(d);result.push_back(d);}
    return result;
}
size_t Count(const std::vector<int64_t>& shape){size_t c=1;for(auto d:shape)c*=size_t(d);return c;}
std::vector<int64_t> Dimensions(const X3TensorInfo& info){return {info.shape,info.shape+info.rank};}
void Require(const X3TensorInfo& info,X3TensorDType type){
    if(info.symbolic||info.device_type!=TensorHelper::CudaDevice||info.dtype!=type||!info.data||info.readonly)
        throw X::Error("CUDA I/O requires writable device tensors with the declared dtype");
    int64_t stride=TensorHelper::ItemSize(type);
    for(int i=int(info.rank)-1;i>=0;--i){
        if(info.shape[i]<=0||(info.strides&&info.strides[i]!=stride))throw X::Error("CUDA I/O requires contiguous positive shapes");
        if(stride>INT64_MAX/info.shape[i])throw X::Error("CUDA tensor byte count overflow");stride*=info.shape[i];}
}
size_t LogicalBytes(const X3TensorInfo& info){
    // SDK byte_size is accessible backing storage, including the remainder
    // after a view's offset. Contiguous operations touch logical elements only.
    uint64_t bytes=TensorHelper::ItemSize(info.dtype);
    for(uint32_t i=0;i<info.rank;++i){
        if(info.shape[i]<=0||uint64_t(info.shape[i])>SIZE_MAX/bytes)
            throw X::Error("CUDA logical tensor byte count overflow");
        bytes*=uint64_t(info.shape[i]);
    }
    if(bytes>info.byte_size)throw X::Error("CUDA logical tensor exceeds backing storage");
    return size_t(bytes);
}
bool SameShape(const X::Tensor& tensor,const std::vector<int64_t>& shape){
    return tensor.IsValid()&&Dimensions(tensor.Info())==shape;
}
}
CUDATensorIO::~CUDATensorIO(){Close();}
void CUDATensorIO::CloseUnlocked(){table_=X::Tensor();headInput_=X::Tensor();headOutput_=X::Tensor();tokenHeadInput_=X::Tensor();tokenHeadOutput_=X::Tensor();
    untiedHead_=packedHead_=packedHeadScales_=X::Tensor();packedHeadScale2_=packedHeadInputScale_=0;nativeNVFP4Head_=false;
    nativeHeadTiledScales_=nativeHeadPackedInput_=nativeHeadInputScales_=nativeHeadWorkspace_=X::Tensor();nativeHeadContext_.reset();
    candidateScratch_=candidateOutput_=indexedScores_=indexedIds_=indexedValues_=X::Tensor();projectionWeights_.clear();projectionWeightGroupMembers_.clear();projectionWeightGroups_.clear();projection_.reset();vocab_=width_=0;}
void CUDATensorIO::Close(){std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();}
bool CUDATensorIO::Open(const std::string& checkpoint){
    std::lock_guard<std::mutex> lock(mutex_);CloseUnlocked();std::string error;
    if(!index_.Open(checkpoint,error))throw X::Error(error);
    auto* entry=index_.Find("model.language_model.embed_tokens.weight");
    if(!entry||entry->dataType!="BF16"||entry->shape.size()!=2||entry->shape[0]<=0||entry->shape[1]<=0||
        entry->shape[0]>INT_MAX||entry->shape[1]>INT_MAX||uint64_t(entry->shape[0])*uint64_t(entry->shape[1])*2!=entry->dataSize)
        throw X::Error("Qwen3.5 tied embedding requires a valid BF16 matrix");
    // A present output head is authoritative. Reject unsupported packed layouts
    // before allocating the embedding; silently tying it changes model logits.
    const auto* head=index_.Find("lm_head.weight");
    const auto checkpointPath=std::filesystem::path(checkpoint);
    const auto configPath=(std::filesystem::is_directory(checkpointPath)?checkpointPath:checkpointPath.parent_path())/"config.json";
    std::ifstream configFile(configPath);
    bool modelopt=false;
    if(configFile){
        nlohmann::json config;
        try{configFile>>config;}catch(...){throw X::Error("CUDA checkpoint config.json is invalid");}
        if(config.contains("quantization_config")&&config["quantization_config"].is_object()){
            const auto& q=config["quantization_config"];
            modelopt=q.value("quant_method",std::string())=="modelopt"&&
                (q.value("quant_algo",std::string())=="MIXED_PRECISION"||q.value("quant_algo",std::string())=="NVFP4");
        }
        auto declaresUntied=[](const nlohmann::json& value){
            return value.is_object()&&value.contains("tie_word_embeddings")&&value["tie_word_embeddings"].is_boolean()&&!value["tie_word_embeddings"].get<bool>();
        };
        if(!head&&(declaresUntied(config)||(config.contains("text_config")&&declaresUntied(config["text_config"]))))
            throw X::Error("CUDA untied checkpoint is missing lm_head.weight");
    }
    const bool packed=head&&head->dataType=="U8"&&modelopt;
    const SafeTensorMetadata* scales=nullptr;
    SafeTensorsMappedFile scalesMapping;
    const void* scaleBytes=nullptr;float headScale2=0,headInputScale=0;
    if(packed){
        const auto rows=entry->shape[0],width=entry->shape[1];
        scales=index_.Find("lm_head.weight_scale");
        const auto* scale2=index_.Find("lm_head.weight_scale_2");
        const auto* inputScale=index_.Find("lm_head.input_scale");
        auto scalar=[&](const SafeTensorMetadata* v){
            if(!v||v->dataType!="F32"||!v->shape.empty()||v->dataSize!=4)throw X::Error("Invalid NVFP4 head scalar scale");
            SafeTensorsMappedFile map;if(!map.Open(v->filePath,error))throw X::Error(error);
            const auto* p=map.DataAt(v->dataOffset,4);if(!p)throw X::Error("NVFP4 scale exceeds checkpoint");
            float value;std::memcpy(&value,p,4);
            if(!std::isfinite(value)||value<=0)throw X::Error("NVFP4 scale must be finite and positive");return value;
        };
        if(width%16||head->shape!=std::vector<long long>{rows,width/2}||head->dataSize!=uint64_t(rows)*width/2||
            !scales||(scales->dataType!="F8_E4M3"&&scales->dataType!="F8_E4M3FN")||
            scales->shape!=std::vector<long long>{rows,width/16}||scales->dataSize!=uint64_t(rows)*width/16)
            throw X::Error("Invalid ModelOpt NVFP4 head layout");
        headScale2=scalar(scale2);headInputScale=scalar(inputScale);
        if(!scalesMapping.Open(scales->filePath,error))throw X::Error(error);
        scaleBytes=scalesMapping.DataAt(scales->dataOffset,scales->dataSize);
        if(!scaleBytes)throw X::Error("NVFP4 block scale storage exceeds checkpoint");
        const auto* codes=static_cast<const unsigned char*>(scaleBytes);
        for(uint64_t i=0;i<scales->dataSize;++i)if(codes[i]>126)throw X::Error("NVFP4 block scale must be nonnegative finite E4M3");
    }
    if(head&&!packed&&(head->dataType!="BF16"||head->shape!=entry->shape||head->dataSize!=entry->dataSize))
        throw X::Error("Separate CUDA output head requires matching BF16 shape; packed heads need quantized head integration");
    SafeTensorsMappedFile mapping;if(!mapping.Open(entry->filePath,error))throw X::Error(error);
    auto* bytes=mapping.DataAt(entry->dataOffset,entry->dataSize);if(!bytes)throw X::Error("embedding storage exceeds checkpoint");
    auto table=TensorHelper::CreateGPU(Host(),X3_TENSOR_BFLOAT16,entry->shape,bytes);
    X::Tensor separateHead,packedWeight,packedScales;
    if(head){
        SafeTensorsMappedFile headMapping;if(!headMapping.Open(head->filePath,error))throw X::Error(error);
        const auto* headBytes=headMapping.DataAt(head->dataOffset,head->dataSize);
        if(!headBytes)throw X::Error("Separate output head storage exceeds checkpoint");
        if(packed){
            packedWeight=TensorHelper::CreateGPU(Host(),X3_TENSOR_UINT8,head->shape,headBytes);
            packedScales=TensorHelper::CreateGPU(Host(),X3_TENSOR_UINT8,scales->shape,scaleBytes);
        }else separateHead=TensorHelper::CreateGPU(Host(),X3_TENSOR_BFLOAT16,head->shape,headBytes);
    }
    auto projection=CreateBF16ProjectionContext();if(!projection)throw X::Error("CUDA tied head context initialization failed");
    table_=std::move(table);untiedHead_=std::move(separateHead);packedHead_=std::move(packedWeight);
    packedHeadScales_=std::move(packedScales);packedHeadScale2_=headScale2;packedHeadInputScale_=headInputScale;
    projection_=std::move(projection);vocab_=int(entry->shape[0]);width_=int(entry->shape[1]);return true;
}
void CUDATensorIO::EnableNVFP4Head(bool enabled){
    std::lock_guard<std::mutex> lock(mutex_);
    if(!enabled){nativeNVFP4Head_=false;nativeHeadContext_.reset();
        nativeHeadTiledScales_=nativeHeadPackedInput_=nativeHeadInputScales_=nativeHeadWorkspace_=X::Tensor();return;}
    if(nativeNVFP4Head_)return;
    if(!packedHead_.IsValid()||vocab_%16||width_%64||
        !std::isfinite(packedHeadScale2_*packedHeadInputScale_)||packedHeadScale2_*packedHeadInputScale_<=0)
        throw X::Error("Native NVFP4 head requires a valid packed block16 checkpoint with rows%16/K%64");
    auto context=CreateModelOptNVFP4Context();if(!context)throw X::Error("Native NVFP4 head requires a supported Blackwell CUDA device");
    const auto scaleBytes=ModelOptNVFP4ScaleBytes(vocab_,width_);
    if(!scaleBytes||scaleBytes>INT_MAX)throw X::Error("Native NVFP4 head scale storage exceeds admitted size");
    auto tiled=TensorHelper::CreateGPU(Host(),X3_TENSOR_UINT8,{int64_t(scaleBytes)});
    auto use=TensorHelper::AcquireGPU({{packedHeadScales_,X3_TENSOR_READ},{tiled,X3_TENSOR_WRITE}});
    Check(RepackModelOptNVFP4Scales(static_cast<const unsigned char*>(packedHeadScales_.Info().data),
        static_cast<unsigned char*>(tiled.Info().data),vocab_,width_,CurrentExecutionStream()));use.Finish();
    nativeHeadTiledScales_=std::move(tiled);nativeHeadContext_=std::move(context);nativeNVFP4Head_=true;
}
X::Value CUDATensorIO::Zeros(X::Value shape){return TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,Shape(shape));}
X::Value CUDATensorIO::StorageView(X::Value source,X::Value dimensions){
    X::Tensor value(source);auto info=value.Info();Require(info,X3_TENSOR_FLOAT32);auto shape=Shape(dimensions);
    if(Count(shape)>Count(Dimensions(info)))throw X::Error("Storage view exceeds source logical capacity");
    std::vector<int64_t> strides(shape.size());int64_t stride=4;
    for(size_t i=shape.size();i-->0;){strides[i]=stride;stride*=shape[i];}
    return value.View(shape,strides);
}
X::Value CUDATensorIO::FiniteF32Bytes(X::Value source){
    // Explicit, bounded diagnostic observation. Never used for model timing.
    static_assert(sizeof(float)==4&&std::numeric_limits<float>::is_iec559,"FP32 bytes require IEEE754");
    const uint32_t endian=1;
    if(*reinterpret_cast<const unsigned char*>(&endian)!=1)throw X::Error("FP32 diagnostic bytes require little-endian host");
    X::Tensor value(source);auto info=value.Info();Require(info,X3_TENSOR_FLOAT32);
    size_t count=1;
    for(uint32_t i=0;i<info.rank;++i){
        if(size_t(info.shape[i])>(1048576U/count))throw X::Error("FP32 diagnostic copy exceeds1048576words");
        count*=size_t(info.shape[i]);
    }
    if(info.byte_size<count*sizeof(float))throw X::Error("FP32 diagnostic source storage is too small");
    std::vector<float> data(count);
    auto use=TensorHelper::AcquireGPU(value,X3_TENSOR_READ);
    Check(cudaMemcpyAsync(data.data(),info.data,count*sizeof(float),cudaMemcpyDeviceToHost,CurrentExecutionStream()));
    Check(cudaStreamSynchronize(CurrentExecutionStream()));use.Finish();
    for(float word:data)if(!std::isfinite(word))throw X::Error("Nonfinite FP32 diagnostic state");
    return X::Value::Bytes(Host(),data.data(),count*sizeof(float));
}
std::string CUDATensorIO::WeightInfo(const std::string& name){
    std::lock_guard<std::mutex> lock(mutex_);auto* entry=index_.Find(name);
    if(!table_.IsValid()||!entry)throw X::Error("Weight metadata requires an open checkpoint and existing name");
    return nlohmann::json{{"shape",entry->shape},{"dtype",entry->dataType},{"bytes",entry->dataSize},{"resident_in_projection_pool",projectionWeights_.find(name)!=projectionWeights_.end()}}.dump();
}
X::Value CUDATensorIO::Weight(const std::string& name){
    std::lock_guard<std::mutex> lock(mutex_);
    if(!table_.IsValid())throw X::Error("Projection weight pool is closed");
    if(name.rfind("model.language_model.layers.",0)!=0&&name.rfind("mtp.",0)!=0)
        throw X::Error("Projection pool admits text-layer/MTP matrices; embedding/head uses its one existing table");
    auto existing=projectionWeights_.find(name);if(existing!=projectionWeights_.end())return existing->second;
    auto* entry=index_.Find(name);
    if(!entry||entry->dataType!="BF16"||entry->shape.size()!=2||entry->shape[0]<1||entry->shape[1]<1||
        entry->shape[0]>INT_MAX||entry->shape[1]>INT_MAX||uint64_t(entry->shape[0])*uint64_t(entry->shape[1])*2!=entry->dataSize)
        throw X::Error("Projection pool requires a valid BF16 matrix");
    SafeTensorsMappedFile mapping;std::string error;if(!mapping.Open(entry->filePath,error))throw X::Error(error);
    auto* data=mapping.DataAt(entry->dataOffset,entry->dataSize);if(!data)throw X::Error("Projection checkpoint storage out of range");
    auto owner=std::make_shared<X::Tensor>(TensorHelper::CreateGPU(Host(),X3_TENSOR_BFLOAT16,entry->shape,data));
    auto info=owner->Info();info.readonly=1;
    auto view=TensorHelper::WrapBorrowedGPU(Host(),info,owner);projectionWeights_.emplace(name,view);return view;
}
void CUDATensorIO::Zero(X::Value tensor){
    X::Tensor dst(tensor);auto info=dst.Info();Require(info,X3_TENSOR_FLOAT32);
    auto use=TensorHelper::AcquireGPU(dst,X3_TENSOR_WRITE);
    Check(cudaMemsetAsync(info.data,0,LogicalBytes(info),CurrentExecutionStream()));use.Finish();
}
X::Value CUDATensorIO::Clone(X::Value value){
    X::Tensor source(value);auto info=source.Info();Require(info,info.dtype);
    LogicalBytes(info);
    auto result=TensorHelper::CreateGPU(Host(),info.dtype,Dimensions(info));CopyInto(source,result);return result;
}
void CUDATensorIO::CopyInto(X::Value source,X::Value destination){
    X::Tensor src(source),dst(destination);auto a=src.Info(),b=dst.Info();Require(a,a.dtype);Require(b,a.dtype);
    if(Dimensions(a)!=Dimensions(b))throw X::Error("CUDA copy requires identical logical shapes");
    const size_t bytes=LogicalBytes(a);
    if(bytes!=LogicalBytes(b))throw X::Error("CUDA copy logical byte counts differ");
    auto pa=reinterpret_cast<uintptr_t>(a.data),pb=reinterpret_cast<uintptr_t>(b.data);
    if(pa!=pb&&(pa>pb?pa-pb:pb-pa)<bytes)throw X::Error("CUDA copy rejects partially overlapping buffers");
    auto use=TensorHelper::AcquireGPU({{src,X3_TENSOR_READ},{dst,X3_TENSOR_WRITE}});
    if(pa!=pb)Check(cudaMemcpyAsync(b.data,a.data,bytes,cudaMemcpyDeviceToDevice,CurrentExecutionStream()));use.Finish();
}
void CUDATensorIO::CopyActive(X::Value source,X::Value active,X::Value destination){
    X::Tensor src(source),flags(active),dst(destination);auto a=src.Info(),b=flags.Info(),c=dst.Info();
    Require(a,X3_TENSOR_FLOAT32);Require(b,X3_TENSOR_FLOAT32);Require(c,X3_TENSOR_FLOAT32);
    if(a.rank!=2||b.rank!=1||Dimensions(a)!=Dimensions(c)||b.shape[0]!=a.shape[0]||
        a.shape[0]>INT_MAX||a.shape[1]>INT_MAX)throw X::Error("active CUDA copy shape mismatch");
    auto use=TensorHelper::AcquireGPU({{src,X3_TENSOR_READ},{flags,X3_TENSOR_READ},{dst,X3_TENSOR_WRITE}});
    Check(CopyQwen35ActiveRows(static_cast<const float*>(a.data),static_cast<const float*>(b.data),
        static_cast<float*>(c.data),a.shape[0],a.shape[1],CurrentExecutionStream()));use.Finish();
}
void CUDATensorIO::WriteChunkToken(X::Value source,X::Value destination,int64_t token){
    X::Tensor src(source),dst(destination);auto a=src.Info(),b=dst.Info();Require(a,X3_TENSOR_FLOAT32);Require(b,X3_TENSOR_FLOAT32);
    if(a.rank!=2||b.rank!=3||a.shape[0]!=b.shape[0]||a.shape[1]!=b.shape[2]||token<0||token>=b.shape[1])
        throw X::Error("CUDA chunk token shape/index invalid");
    auto use=TensorHelper::AcquireGPU({{src,X3_TENSOR_READ},{dst,X3_TENSOR_WRITE}});
    Check(cudaMemcpy2DAsync(static_cast<float*>(b.data)+token*b.shape[2],b.shape[1]*b.shape[2]*4,
        a.data,a.shape[1]*4,a.shape[1]*4,a.shape[0],cudaMemcpyDeviceToDevice,CurrentExecutionStream()));use.Finish();
}
void CUDATensorIO::CopyLastChunk(X::Value source,X::Value active,X::Value destination){
    X::Tensor src(source),flags(active),dst(destination);auto a=src.Info(),b=flags.Info(),c=dst.Info();
    Require(a,X3_TENSOR_FLOAT32);Require(b,X3_TENSOR_FLOAT32);Require(c,X3_TENSOR_FLOAT32);
    if(a.rank!=3||b.rank!=2||c.rank!=2||a.shape[0]!=b.shape[0]||a.shape[1]!=b.shape[1]||a.shape[0]!=c.shape[0]||a.shape[2]!=c.shape[1])
        throw X::Error("CUDA last chunk shape mismatch");
    auto use=TensorHelper::AcquireGPU({{src,X3_TENSOR_READ},{flags,X3_TENSOR_READ},{dst,X3_TENSOR_WRITE}});
    Check(CopyQwen35LastChunk(static_cast<const float*>(a.data),static_cast<const float*>(b.data),static_cast<float*>(c.data),
        a.shape[0],a.shape[1],a.shape[2],CurrentExecutionStream()));use.Finish();
}
X::Value CUDATensorIO::FromHost(X::Value values,X::Value dimensions){
    auto shape=Shape(dimensions);if(!values.IsList()||values.Size()!=Count(shape))throw X::Error("CUDA input element count mismatch");
    std::vector<float> data(values.Size());for(uint64_t i=0;i<values.Size();++i)data[i]=float(values.Get(i).ToDouble());
    return TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,shape,data.data());
}
void CUDATensorIO::MaskChunk(X::Value hidden,X::Value active){
    X::Tensor value(hidden),flags(active);auto a=value.Info(),b=flags.Info();Require(a,X3_TENSOR_FLOAT32);Require(b,X3_TENSOR_FLOAT32);
    if(a.rank!=3||b.rank!=2||a.shape[0]!=b.shape[0]||a.shape[1]!=b.shape[1])throw X::Error("Prefix mask dimensions invalid");
    auto use=TensorHelper::AcquireGPU({{value,X3_TENSOR_WRITE},{flags,X3_TENSOR_READ}});
    Check(MaskQwen35Chunk(static_cast<float*>(a.data),static_cast<const float*>(b.data),a.shape[0],a.shape[1],a.shape[2],CurrentExecutionStream()));use.Finish();
}
void CUDATensorIO::RestoreKVSuffix(X::Value cache,X::Value backup,X::Value spans){
    X::Tensor value(cache),old(backup),meta(spans);auto a=value.Info(),b=old.Info(),c=meta.Info();
    Require(a,X3_TENSOR_FLOAT32);Require(b,X3_TENSOR_FLOAT32);Require(c,X3_TENSOR_FLOAT32);
    if(a.rank!=4||Dimensions(a)!=Dimensions(b)||c.rank!=2||c.shape[0]!=a.shape[0]||c.shape[1]!=2)throw X::Error("Prefix KV dimensions invalid");
    auto use=TensorHelper::AcquireGPU({{value,X3_TENSOR_WRITE},{old,X3_TENSOR_READ},{meta,X3_TENSOR_READ}});
    Check(RestoreQwen35KVSuffix(static_cast<float*>(a.data),static_cast<const float*>(b.data),static_cast<const float*>(c.data),a.shape[0],a.shape[1],a.shape[2],a.shape[3],CurrentExecutionStream()));use.Finish();
}
void CUDATensorIO::Update(X::Value destination,X::Value values){
    X::Tensor dst(destination);auto info=dst.Info();Require(info,X3_TENSOR_FLOAT32);
    if(!values.IsList()||values.Size()!=Count(Dimensions(info)))throw X::Error("CUDA update element count mismatch");
    std::vector<float> data(values.Size());for(uint64_t i=0;i<values.Size();++i)data[i]=float(values.Get(i).ToDouble());
    auto use=TensorHelper::AcquireGPU(dst,X3_TENSOR_WRITE);
    Check(cudaMemcpyAsync(info.data,data.data(),data.size()*4,cudaMemcpyHostToDevice,CurrentExecutionStream()));
    Check(cudaStreamSynchronize(CurrentExecutionStream()));use.Finish();
}
void CUDATensorIO::UpdateRequest(X::Value active,X::Value positions,X::Value slots,X::Value flags,X::Value positionValues,X::Value slotValues){
    X::Tensor a(active),p(positions),s(slots);auto ai=a.Info(),pi=p.Info(),si=s.Info();
    Require(ai,X3_TENSOR_FLOAT32);Require(pi,X3_TENSOR_FLOAT32);Require(si,X3_TENSOR_FLOAT32);
    if(ai.rank!=2||pi.rank!=3||Dimensions(ai)!=Dimensions(si)||pi.shape[0]!=ai.shape[0]||pi.shape[1]!=ai.shape[1]||pi.shape[2]!=3||
        (ai.shape[0]!=1&&ai.shape[0]!=2&&ai.shape[0]!=4&&ai.shape[0]!=8)||ai.shape[1]>128)throw X::Error("Request controls require bounded batch/time shapes");
    const int count=int(ai.shape[0]*ai.shape[1]);
    if(!flags.IsList()||!positionValues.IsList()||!slotValues.IsList()||flags.Size()!=count||positionValues.Size()!=3*count||slotValues.Size()!=count)
        throw X::Error("Request control element counts differ");
    auto overlaps=[](const X3TensorInfo& x,const X3TensorInfo& y){auto a=reinterpret_cast<uintptr_t>(x.data),b=reinterpret_cast<uintptr_t>(y.data);return a<=b?b-a<x.byte_size:a-b<y.byte_size;};
    if(overlaps(ai,pi)||overlaps(ai,si)||overlaps(pi,si))throw X::Error("Request control outputs must not overlap");
    std::vector<float> f(count),pos(3*count),indices(count);
    for(int i=0;i<count;++i){f[i]=float(flags.Get(i).ToDouble());indices[i]=float(slotValues.Get(i).ToDouble());for(int axis=0;axis<3;++axis)pos[3*i+axis]=float(positionValues.Get(3*i+axis).ToDouble());}
    auto use=TensorHelper::AcquireGPU({{a,X3_TENSOR_WRITE},{p,X3_TENSOR_WRITE},{s,X3_TENSOR_WRITE}});
    Check(WriteQwen35Request((float*)ai.data,(float*)pi.data,(float*)si.data,f.data(),pos.data(),indices.data(),count,CurrentExecutionStream()));use.Finish();
}
void CUDATensorIO::EmbeddingChunk(X::Value tokens,X::Value destination){
    std::lock_guard<std::mutex> lock(mutex_);X::Tensor dst(destination);auto info=dst.Info();Require(info,X3_TENSOR_FLOAT32);
    if(!table_.IsValid()||info.rank!=3||(info.shape[0]!=1&&info.shape[0]!=2&&info.shape[0]!=4&&info.shape[0]!=8)||info.shape[1]>128||info.shape[2]!=width_||
        !tokens.IsList()||tokens.Size()<1||tokens.Size()>info.shape[1])throw X::Error("Chunk embedding requires bounded batch/time/hidden shape");
    std::vector<int> ids;ids.reserve(tokens.Size()*info.shape[0]);
    for(uint64_t t=0;t<tokens.Size();++t){auto row=tokens.Get(t);if(!row.IsList()||row.Size()!=info.shape[0])throw X::Error("Chunk embedding token batch differs");
        for(uint64_t b=0;b<row.Size();++b){auto id=row.Get(b).ToInt64();if(id<0||id>=vocab_)throw X::Error("Chunk embedding token out of range");ids.push_back(int(id));}}
    auto table=table_.Info();auto start=reinterpret_cast<uintptr_t>(table.data),end=reinterpret_cast<uintptr_t>(info.data);
    if(start<=end?end-start<table.byte_size:start-end<info.byte_size)throw X::Error("Embedding destination must not overlap table");
    auto use=TensorHelper::AcquireGPU({{table_,X3_TENSOR_READ},{dst,X3_TENSOR_WRITE}});
    Check(GatherQwen35Chunk(table.data,(float*)info.data,ids.data(),info.shape[0],info.shape[1],tokens.Size(),width_,CurrentExecutionStream()));use.Finish();
}
X::Value CUDATensorIO::ReadF32(const std::string& path,X::Value dimensions){
    auto shape=Shape(dimensions);std::vector<float> data(Count(shape));
    std::ifstream f(path,std::ios::binary|std::ios::ate);
    if(!f||f.tellg()!=std::streamoff(data.size()*4))throw X::Error("FP32 fixture byte count mismatch");
    f.seekg(0);f.read(reinterpret_cast<char*>(data.data()),data.size()*4);if(!f)throw X::Error("FP32 input read failed");
    return TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,shape,data.data());
}
X::Value CUDATensorIO::Embedding(X::Value tokens){
    std::lock_guard<std::mutex> lock(mutex_);
    if(!table_.IsValid()||!tokens.IsList()||tokens.Size()<1||tokens.Size()>8192)throw X::Error("embedding requires an open model and 1..8192 IDs");
    std::vector<long long> ids(tokens.Size());for(uint64_t i=0;i<tokens.Size();++i){ids[i]=tokens.Get(i).ToInt64();if(ids[i]<0||ids[i]>=vocab_)throw X::Error("embedding token out of range");}
    auto indices=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{int64_t(ids.size())},ids.data());
    auto output=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{int64_t(ids.size()),width_});
    auto use=TensorHelper::AcquireGPU({{table_,X3_TENSOR_READ},{indices,X3_TENSOR_READ},{output,X3_TENSOR_WRITE}});
    Check(runEmbeddingGatherInt64BF16ToFP32(static_cast<const bfloat16*>(table_.Info().data),
        static_cast<const long long*>(indices.Info().data),static_cast<float*>(output.Info().data),ids.size(),vocab_,width_,CurrentExecutionStream()));
    use.Finish();return output;
}
void CUDATensorIO::SliceInto(X::Value source,int64_t start,X::Value destination){
    X::Tensor src(source),dst(destination);auto a=src.Info(),b=dst.Info();Require(a,X3_TENSOR_FLOAT32);Require(b,X3_TENSOR_FLOAT32);
    if(a.rank!=2||b.rank<1||a.shape[0]!=b.shape[0]||start<0)throw X::Error("slice requires packed rows and matching batch");
    size_t width=Count(Dimensions(b))/size_t(b.shape[0]);
    if(start>a.shape[1]||width>size_t(a.shape[1]-start))throw X::Error("packed CUDA slice out of range");
    auto use=TensorHelper::AcquireGPU({{src,X3_TENSOR_READ},{dst,X3_TENSOR_WRITE}});
    Check(cudaMemcpy2DAsync(b.data,width*4,static_cast<float*>(a.data)+start,size_t(a.shape[1])*4,
        width*4,size_t(b.shape[0]),cudaMemcpyDeviceToDevice,CurrentExecutionStream()));use.Finish();
}
X::Value CUDATensorIO::Slice(X::Value source,int64_t start,X::Value shape){
    auto result=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,Shape(shape));SliceInto(source,start,result);return result;
}
X::Value CUDATensorIO::VisualToken(X::Value source,int64_t token){
    X::Tensor src(source);auto a=src.Info();Require(a,X3_TENSOR_FLOAT32);
    if(a.rank!=3||token<0||token>=a.shape[1])throw X::Error("visual token index/rank invalid");
    auto output=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{a.shape[0],a.shape[2]});
    auto use=TensorHelper::AcquireGPU({{src,X3_TENSOR_READ},{output,X3_TENSOR_WRITE}});
    Check(cudaMemcpy2DAsync(output.Info().data,a.shape[2]*4,static_cast<float*>(a.data)+token*a.shape[2],
        a.shape[1]*a.shape[2]*4,a.shape[2]*4,a.shape[0],cudaMemcpyDeviceToDevice,CurrentExecutionStream()));use.Finish();return output;
}
X::Value CUDATensorIO::Row(X::Value source,int64_t row){
    X::Tensor src(source);auto a=src.Info();Require(a,X3_TENSOR_FLOAT32);
    if(a.rank!=2||row<0||row>=a.shape[0])throw X::Error("CUDA row index/rank invalid");
    auto output=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{1,a.shape[1]});
    auto use=TensorHelper::AcquireGPU({{src,X3_TENSOR_READ},{output,X3_TENSOR_WRITE}});
    Check(cudaMemcpyAsync(output.Info().data,static_cast<float*>(a.data)+row*a.shape[1],a.shape[1]*4,cudaMemcpyDeviceToDevice,CurrentExecutionStream()));
    use.Finish();return output;
}
X::Value CUDATensorIO::StackRows(X::Value rows){
    if(!rows.IsList()||rows.Size()<1||rows.Size()>8)throw X::Error("CUDA stacked batch requires 1..8 rows");
    X::Tensor first(rows.Get(uint64_t(0)));auto a=first.Info();Require(a,X3_TENSOR_FLOAT32);
    if(a.rank!=2||a.shape[0]!=1)throw X::Error("CUDA stacked rows require [1,hidden] tensors");
    if(rows.Size()==1)return first;
    auto output=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{int64_t(rows.Size()),a.shape[1]});
    for(uint64_t i=0;i<rows.Size();++i){
        X::Tensor row(rows.Get(i));auto b=row.Info();Require(b,X3_TENSOR_FLOAT32);
        if(Dimensions(a)!=Dimensions(b))throw X::Error("CUDA stacked row dimensions mismatch");
        auto use=TensorHelper::AcquireGPU({{row,X3_TENSOR_READ},{output,X3_TENSOR_WRITE}});
        Check(cudaMemcpyAsync(static_cast<float*>(output.Info().data)+i*a.shape[1],b.data,a.shape[1]*4,cudaMemcpyDeviceToDevice,CurrentExecutionStream()));use.Finish();
    }
    return output;
}
X::Value CUDATensorIO::LogitsUnlocked(X::Value hidden,bool reuse,bool tokens){
    X::Tensor input(hidden);auto a=input.Info();Require(a,X3_TENSOR_FLOAT32);
    if(!table_.IsValid()||(a.rank!=2&&!(tokens&&a.rank==3))||a.shape[a.rank-1]!=width_)
        throw X::Error("tied head input/model shape mismatch");
    int64_t rows=a.shape[0];
    if(a.rank==3){if(rows>128/a.shape[1])throw X::Error("token head exceeds128rows");rows*=a.shape[1];}
    if(rows>(tokens?128:8)||rows>INT_MAX/width_)throw X::Error("tied head row/element limit exceeded");
    std::vector<int64_t> inShape{rows,width_},outShape{rows,vocab_};
    auto& inputBuffer=tokens?tokenHeadInput_:headInput_;auto& outputBuffer=tokens?tokenHeadOutput_:headOutput_;
    if(!SameShape(inputBuffer,inShape))inputBuffer=TensorHelper::CreateGPU(Host(),X3_TENSOR_BFLOAT16,inShape);
    X::Tensor output;
    if(reuse){if(!SameShape(outputBuffer,outShape))outputBuffer=TensorHelper::CreateGPU(Host(),X3_TENSOR_BFLOAT16,outShape);output=outputBuffer;}
    else output=TensorHelper::CreateGPU(Host(),X3_TENSOR_BFLOAT16,outShape);
    const bool packed=packedHead_.IsValid();
    std::vector<std::pair<X::Tensor,X3TensorAccess>> accesses{{input,X3_TENSOR_READ},{packed?packedHead_:HeadTable(),X3_TENSOR_READ},
        {inputBuffer,X3_TENSOR_WRITE},{output,X3_TENSOR_WRITE}};
    if(packed)accesses.push_back({packedHeadScales_,X3_TENSOR_READ});
    if(nativeNVFP4Head_){
        const std::vector<int64_t> packedShape{rows,width_/2},scaleShape{int64_t(ModelOptNVFP4ScaleBytes(int(rows),width_))};
        if(!SameShape(nativeHeadPackedInput_,packedShape))nativeHeadPackedInput_=TensorHelper::CreateGPU(Host(),X3_TENSOR_UINT8,packedShape);
        if(!SameShape(nativeHeadInputScales_,scaleShape))nativeHeadInputScales_=TensorHelper::CreateGPU(Host(),X3_TENSOR_UINT8,scaleShape);
        if(!nativeHeadWorkspace_.IsValid())nativeHeadWorkspace_=TensorHelper::CreateGPU(Host(),X3_TENSOR_UINT8,{4LL<<20});
        accesses.push_back({nativeHeadTiledScales_,X3_TENSOR_READ});accesses.push_back({nativeHeadPackedInput_,X3_TENSOR_WRITE});
        accesses.push_back({nativeHeadInputScales_,X3_TENSOR_WRITE});accesses.push_back({nativeHeadWorkspace_,X3_TENSOR_WRITE});
    }
    auto use=TensorHelper::AcquireGPU(accesses);
    Check(runConvertFP32ToBF16Async(static_cast<const float*>(a.data),static_cast<bfloat16*>(inputBuffer.Info().data),rows*width_,CurrentExecutionStream()));
    if(nativeNVFP4Head_){
        Check(QuantizeModelOptNVFP4Activation(inputBuffer.Info().data,static_cast<unsigned char*>(nativeHeadPackedInput_.Info().data),
            static_cast<unsigned char*>(nativeHeadInputScales_.Info().data),packedHeadInputScale_,int(rows),width_,CurrentExecutionStream()));
        const auto status=RunModelOptNVFP4MatmulContext(nativeHeadContext_,packedHead_.Info().data,nativeHeadTiledScales_.Info().data,
            nativeHeadPackedInput_.Info().data,nativeHeadInputScales_.Info().data,output.Info().data,int(rows),vocab_,width_,
            packedHeadScale2_*packedHeadInputScale_,nativeHeadWorkspace_.Info().data,4ULL<<20,CurrentExecutionStream());
        if(status!=CUBLAS_STATUS_SUCCESS)throw X::Error("Native NVFP4 head cuBLASLt failure: "+std::to_string(int(status)));
    }else if(packed)Check(RunModelOptNVFP4Projection(inputBuffer.Info().data,
        static_cast<const unsigned char*>(packedHead_.Info().data),static_cast<const unsigned char*>(packedHeadScales_.Info().data),
        packedHeadScale2_,output.Info().data,int(rows),vocab_,width_,CurrentExecutionStream()));
    else if(RunBF16Projection(projection_,inputBuffer.Info().data,HeadTable().Info().data,output.Info().data,rows,width_,vocab_,CurrentExecutionStream()))
        throw X::Error("CUDA tied head projection failed");
    use.Finish();return output;
}
X::Value CUDATensorIO::Logits(X::Value hidden){std::lock_guard<std::mutex> lock(mutex_);return LogitsUnlocked(hidden,false);}
X::Value CUDATensorIO::ArgmaxUnlocked(X::Value logits,int maxRows){
    X::Tensor input(logits);auto a=input.Info();
    if(a.dtype!=X3_TENSOR_FLOAT32&&a.dtype!=X3_TENSOR_BFLOAT16)throw X::Error("argmax requires BF16/FP32 logits");
    Require(a,a.dtype);if(a.rank!=2||a.shape[0]>maxRows||a.shape[1]>INT_MAX)throw X::Error("argmax batch/logits shape invalid");
    auto ids=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,{a.shape[0]});
    auto values=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,{a.shape[0]});
    auto use=TensorHelper::AcquireGPU({{input,X3_TENSOR_READ},{ids,X3_TENSOR_WRITE},{values,X3_TENSOR_WRITE}});
    auto* outIds=static_cast<long long*>(ids.Info().data);auto* outValues=static_cast<float*>(values.Info().data);
    if(a.dtype==X3_TENSOR_BFLOAT16)Check(runLogitsTop1BatchBF16(static_cast<const bfloat16*>(a.data),outIds,outValues,a.shape[0],a.shape[1],CurrentExecutionStream()));
    else Check(runLogitsTop1BatchFP32(static_cast<const float*>(a.data),outIds,outValues,a.shape[0],a.shape[1],CurrentExecutionStream()));
    std::vector<long long> hostIds(a.shape[0]);std::vector<float> hostValues(a.shape[0]);
    Check(cudaMemcpyAsync(hostIds.data(),outIds,hostIds.size()*8,cudaMemcpyDeviceToHost,CurrentExecutionStream()));
    Check(cudaMemcpyAsync(hostValues.data(),outValues,hostValues.size()*4,cudaMemcpyDeviceToHost,CurrentExecutionStream()));
    Check(cudaStreamSynchronize(CurrentExecutionStream()));use.Finish();auto result=X::Value::List(Host());
    for(size_t i=0;i<hostIds.size();++i){if(hostIds[i]<0||!std::isfinite(hostValues[i]))throw X::Error("no finite CUDA head prediction");result.Append(X::Value(int64_t(hostIds[i])));}return result;
}
X::Value CUDATensorIO::Argmax(X::Value logits){std::lock_guard<std::mutex> lock(mutex_);return ArgmaxUnlocked(logits);}
X::Value CUDATensorIO::Predict(X::Value hidden){std::lock_guard<std::mutex> lock(mutex_);return ArgmaxUnlocked(LogitsUnlocked(hidden,true));}
X::Value CUDATensorIO::PredictTokens(X::Value hidden){std::lock_guard<std::mutex> lock(mutex_);return ArgmaxUnlocked(LogitsUnlocked(hidden,true,true),128);}
X::Value CUDATensorIO::HeadCandidates(X::Value selectedRows,int64_t count,bool tokenHead){
    std::lock_guard<std::mutex> lock(mutex_);auto& source=tokenHead?tokenHeadOutput_:headOutput_;
    if(!table_.IsValid()||!source.IsValid()||!selectedRows.IsList()||selectedRows.Size()<1||selectedRows.Size()>8||count<1||count>65536||count>vocab_)
        throw X::Error("CUDA candidate head requires cached target logits,1..8 rows and valid candidate count");
    auto info=source.Info();std::vector<int> rows(selectedRows.Size());
    for(uint64_t i=0;i<selectedRows.Size();++i){auto r=selectedRows.Get(i).ToInt64();if(r<0||r>=info.shape[0])throw X::Error("Candidate source row out of range");rows[i]=int(r);}
    size_t bytes=Qwen35HeadCandidatesWorkspace(vocab_);if(!bytes)throw X::Error("Candidate workspace query failed");
    std::vector<int64_t> shape{int64_t(rows.size()),count},scratchShape{int64_t((bytes+3)/4)};
    if(!SameShape(candidateScratch_,scratchShape))candidateScratch_=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT32,scratchShape);
    if(!SameShape(candidateOutput_,shape))candidateOutput_=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT32,shape);
    auto use=TensorHelper::AcquireGPU({{source,X3_TENSOR_READ},{candidateScratch_,X3_TENSOR_WRITE},{candidateOutput_,X3_TENSOR_WRITE}});
    if(RunQwen35HeadCandidates(rows.size(),info.shape[0],vocab_,count,rows.data(),info.data,
        static_cast<int*>(candidateOutput_.Info().data),candidateScratch_.Info().data,candidateScratch_.Info().byte_size,CurrentExecutionStream()))throw X::Error("CUDA candidate selection failed");
    use.Finish();return candidateOutput_; // Reused at the next candidate selection, no weight copy.
}
X::Value CUDATensorIO::PredictIndexed(X::Value hidden,X::Value candidates){
    std::lock_guard<std::mutex> lock(mutex_);X::Tensor query(hidden),indices(candidates);auto a=query.Info(),b=indices.Info();
    if(packedHead_.IsValid())throw X::Error("Indexed NVFP4 head is not implemented; use full-head prediction");
    Require(a,X3_TENSOR_FLOAT32);Require(b,X3_TENSOR_INT32);
    if(!table_.IsValid()||a.rank!=2||b.rank!=2||a.shape[0]<1||a.shape[0]>8||a.shape[1]!=width_||a.shape[0]!=b.shape[0]||b.shape[1]>65536||b.shape[1]>vocab_)
        throw X::Error("CUDA indexed proposal head shape invalid");
    std::vector<int64_t> scoreShape{a.shape[0],b.shape[1]},rowShape{a.shape[0]};
    if(!SameShape(indexedScores_,scoreShape))indexedScores_=TensorHelper::CreateGPU(Host(),X3_TENSOR_BFLOAT16,scoreShape);
    if(!SameShape(indexedIds_,rowShape))indexedIds_=TensorHelper::CreateGPU(Host(),X3_TENSOR_INT64,rowShape);
    if(!SameShape(indexedValues_,rowShape))indexedValues_=TensorHelper::CreateGPU(Host(),X3_TENSOR_FLOAT32,rowShape);
    auto use=TensorHelper::AcquireGPU({{query,X3_TENSOR_READ},{indices,X3_TENSOR_READ},{HeadTable(),X3_TENSOR_READ},
        {indexedScores_,X3_TENSOR_WRITE},{indexedIds_,X3_TENSOR_WRITE},{indexedValues_,X3_TENSOR_WRITE}});
    Qwen35IndexedHeadConfig c{int(a.shape[0]),width_,vocab_,int(b.shape[1])};
    if(RunQwen35IndexedHead(c,static_cast<const float*>(a.data),HeadTable().Info().data,static_cast<const int*>(b.data),indexedScores_.Info().data,
        static_cast<long long*>(indexedIds_.Info().data),static_cast<float*>(indexedValues_.Info().data),CurrentExecutionStream()))throw X::Error("CUDA indexed proposal head failed");
    std::vector<long long> ids(a.shape[0]);std::vector<float> values(a.shape[0]);
    Check(cudaMemcpyAsync(ids.data(),indexedIds_.Info().data,ids.size()*8,cudaMemcpyDeviceToHost,CurrentExecutionStream()));
    Check(cudaMemcpyAsync(values.data(),indexedValues_.Info().data,values.size()*4,cudaMemcpyDeviceToHost,CurrentExecutionStream()));
    Check(cudaStreamSynchronize(CurrentExecutionStream()));use.Finish();auto result=X::Value::List(Host());
    for(size_t i=0;i<ids.size();++i){if(ids[i]<0||!std::isfinite(values[i]))throw X::Error("No finite indexed proposal");result.Append(X::Value(int64_t(ids[i])));}return result;
}
X::Value CUDATensorIO::PrepareImage(const std::string& path,int pixels){
    if(pixels<1024)throw X::Error("Qwen3.5 image pixel budget must fit a merged 32x32 patch");
    if(pixels<1024||pixels>1048576)throw X::Error("CUDA image pixel budget outside admitted range");
    void* memory=nullptr;size_t bytes=0;long long grid[3]{};int sh=0,sw=0,rh=0,rw=0,patches=0,features=0;char error[1024]{};
    if(GarnetQwenVLPreprocessJpegFileDevice(path.c_str(),1024,pixels,&memory,&bytes,grid,&sh,&sw,&rh,&rw,&patches,&features,error,sizeof(error)))throw X::Error(error);
    try{
        if(grid[0]!=1||grid[1]<=0||grid[2]<=0||patches!=grid[1]*grid[2]||features!=1536||bytes!=size_t(patches)*features*4)
            throw X::Error("CUDA JPEG frontend returned inconsistent Qwen3.5 patches");
        std::vector<int64_t> shape{1,patches,features},strides{int64_t(bytes),features*4,4};X3TensorInfo info{};
        info.size=sizeof(info);info.dtype=X3_TENSOR_FLOAT32;info.rank=3;info.shape=shape.data();info.strides=strides.data();info.data=memory;info.byte_size=bytes;
        int device=0;Check(cudaGetDevice(&device));auto tensor=TensorHelper::WrapGPU(Host(),info,memory,device);memory=nullptr;
        auto result=X::Value::List(Host());result.Append(tensor);result.Append(X::Value(int64_t(grid[1])));result.Append(X::Value(int64_t(grid[2])));return result;
    }catch(...){if(memory)cudaFree(memory);throw;}
}
std::string CUDATensorIO::MemoryJson() const{
    std::lock_guard<std::mutex> lock(mutex_);size_t free=0,total=0;Check(cudaMemGetInfo(&free,&total));
    auto bytes=[](const X::Tensor& t)->uint64_t{return t.IsValid()?t.Info().byte_size:0;};
    uint64_t poolBytes=0,groupBytes=0;
    for(const auto& item:projectionWeightGroups_)groupBytes+=bytes(item.second);
    poolBytes=groupBytes;
    for(const auto& item:projectionWeights_)if(!projectionWeightGroupMembers_.count(item.first))poolBytes+=bytes(item.second);
    return nlohmann::json{{"vocab_size",vocab_},{"hidden_size",width_},{"embedding_head_tied",!untiedHead_.IsValid()&&!packedHead_.IsValid()},
        {"embedding_table_bytes",bytes(table_)},{"untied_head_bytes",bytes(untiedHead_)+bytes(packedHead_)+bytes(packedHeadScales_)},
        {"head_precision",nativeNVFP4Head_?"modelopt_nvfp4_w4a4":(packedHead_.IsValid()?"modelopt_nvfp4_w4a16":"bf16")},
        {"native_head_workspace_bytes",bytes(nativeHeadTiledScales_)+bytes(nativeHeadPackedInput_)+bytes(nativeHeadInputScales_)+bytes(nativeHeadWorkspace_)},
        {"native_head_tiled_weight_scales_bytes",bytes(nativeHeadTiledScales_)},
        {"embedding_head_shared_bytes",untiedHead_.IsValid()||packedHead_.IsValid()?0:bytes(table_)},{"head_input_bytes",bytes(headInput_)+bytes(tokenHeadInput_)},{"head_output_bytes",bytes(headOutput_)+bytes(tokenHeadOutput_)},
        {"single_head_input_bytes",bytes(headInput_)},{"single_head_output_bytes",bytes(headOutput_)},
        {"token_head_input_bytes",bytes(tokenHeadInput_)},{"token_head_output_bytes",bytes(tokenHeadOutput_)},
        {"candidate_workspace_bytes",bytes(candidateScratch_)},{"candidate_index_bytes",bytes(candidateOutput_)},
        {"indexed_head_buffers_bytes",bytes(indexedScores_)+bytes(indexedIds_)+bytes(indexedValues_)},
        {"projection_weight_pool_bytes",poolBytes},{"projection_weight_pool_entries",projectionWeights_.size()},
        {"joined_projection_group_count",projectionWeightGroups_.size()},{"joined_projection_group_bytes",groupBytes},
        {"joined_projection_member_count",projectionWeightGroupMembers_.size()},
        {"device_used_bytes",total-free},{"scope","owned I/O buffers plus whole-device snapshot; engine weights/state and process peak are separate"}}.dump();
}
int64_t CUDATensorIO::TensorBytes(X::Value value){
    X::Tensor tensor(value);auto info=tensor.Info();Require(info,info.dtype);return info.byte_size;
}
}
#endif
