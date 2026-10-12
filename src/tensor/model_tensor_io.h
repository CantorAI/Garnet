// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include "safetensors_index.h"
#include "lowering_context.h"
#include <cstring>
#include <limits>
#include <cmath>
#include <fstream>
namespace Garnet {
// Host-side token lookup and state movement. Neural projections belong to the graph.
class ModelTensorIO {
    SafeTensorsIndex index_;
    SafeTensorsMappedFile mapping_;
    const SafeTensorMetadata* embedding_=nullptr;
    static std::vector<int64_t> Shape(X::Value dimensions) {
        if(!dimensions.IsList()||!dimensions.Size())throw X::Error("shape must be a nonempty list");
        std::vector<int64_t> shape;
        for(uint64_t i=0;i<dimensions.Size();++i){auto d=dimensions.Get(i).ToInt64();
            if(d<=0)throw X::Error("shape dimensions must be positive");shape.push_back(d);}
        return shape;
    }
    static size_t Count(const std::vector<int64_t>& shape) {
        size_t count=1;for(auto d:shape){if(d<=0||uint64_t(d)>SIZE_MAX/count)throw X::Error("tensor shape overflow");count*=size_t(d);}
        if(count>SIZE_MAX/sizeof(float))throw X::Error("tensor byte length overflow");return count;
    }
    static void RequireHost(const X::Tensor& tensor) {
        auto info=tensor.Info();
        if(info.symbolic||info.device_type!=0||info.dtype!=X3_TENSOR_FLOAT32||!info.data||!IsContiguousTensor(tensor))
            throw X::Error("expected contiguous host FP32 tensor");
    }
public:
    void Close(){embedding_=nullptr;mapping_.Close();}
    bool Open(const std::string& path) {
        Close();std::string error;if(!index_.Open(path,error))throw X::Error(error);
        const auto* entry=index_.Find("model.language_model.embed_tokens.weight");
        if(!entry||entry->dataType!="BF16"||entry->shape.size()!=2||!entry->shape[0]||!entry->shape[1]||
            entry->shape[0]>INT64_MAX||entry->shape[1]>INT64_MAX||entry->shape[0]>UINT64_MAX/2/entry->shape[1]||
            entry->shape[0]*entry->shape[1]*2!=entry->dataSize)throw X::Error("invalid BF16 embedding table");
        if(!mapping_.Open(entry->filePath,error))throw X::Error(error);
        if(!mapping_.DataAt(entry->dataOffset,entry->dataSize))throw X::Error("embedding exceeds checkpoint");
        embedding_=entry;return true;
    }
    X::Value Embedding(X::Value tokens) {
        if(!embedding_||!tokens.IsList()||!tokens.Size())throw X::Error("open checkpoint and supply token IDs");
        const auto width=embedding_->shape[1];std::vector<int64_t> shape{int64_t(tokens.Size()),int64_t(width)};
        std::vector<float> values(Count(shape));
        const auto* bytes=static_cast<const unsigned char*>(mapping_.DataAt(embedding_->dataOffset,embedding_->dataSize));
        for(uint64_t b=0;b<tokens.Size();++b){auto id=tokens.Get(b).ToInt64();
            if(id<0||uint64_t(id)>=embedding_->shape[0])throw X::Error("embedding token out of range");
            for(uint64_t d=0;d<width;++d){uint16_t bf16;std::memcpy(&bf16,bytes+2*(uint64_t(id)*width+d),2);
                uint32_t bits=uint32_t(bf16)<<16;std::memcpy(&values[b*width+d],&bits,4);}}
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    X::Value Zeros(X::Value dimensions) {
        auto shape=Shape(dimensions);std::vector<float> values(Count(shape),0);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    X::Value Slice(X::Value source,int64_t start,X::Value dimensions) {
        X::Tensor tensor(source);RequireHost(tensor);auto info=tensor.Info();auto shape=Shape(dimensions);
        const auto count=Count(shape),width=count/size_t(shape[0]);
        if(info.rank!=2||start<0||shape[0]!=info.shape[0]||start>info.shape[1]||width>uint64_t(info.shape[1]-start))
            throw X::Error("packed tensor slice shape/bounds invalid");
        auto lease=tensor.Acquire();auto* input=static_cast<const float*>(info.data);std::vector<float> values(count);
        for(size_t b=0;b<size_t(shape[0]);++b)std::memcpy(values.data()+b*width,input+b*info.shape[1]+start,width*4);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    X::Value ReadF32(const std::string& path,X::Value dimensions) {
        auto shape=Shape(dimensions);std::vector<float> values(Count(shape));
        std::ifstream file(path,std::ios::binary|std::ios::ate);
        if(!file||file.tellg()!=std::streamoff(values.size()*4))throw X::Error("FP32 file length mismatch");
        file.seekg(0);file.read(reinterpret_cast<char*>(values.data()),values.size()*4);
        if(!file)throw X::Error("FP32 file read failed");
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    X::Value VisualToken(X::Value source,int64_t token) {
        X::Tensor tensor(source);RequireHost(tensor);auto info=tensor.Info();
        if(info.rank!=3||token<0||token>=info.shape[1])throw X::Error("visual token index/shape invalid");
        auto lease=tensor.Acquire();const auto* input=static_cast<const float*>(info.data);
        std::vector<int64_t> shape{info.shape[0],info.shape[2]};std::vector<float> values(Count(shape));
        for(int64_t b=0;b<info.shape[0];++b)std::memcpy(values.data()+b*info.shape[2],
            input+(b*info.shape[1]+token)*info.shape[2],info.shape[2]*4);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,values.data(),values.size()*4);
    }
    // Compatibility with the serving caller: these logits are already projected on GPU.
    X::Value Logits(X::Value source) {
        X::Tensor tensor(source);RequireHost(tensor);auto info=tensor.Info();
        if(!embedding_||info.rank!=2||uint64_t(info.shape[1])!=embedding_->shape[0])throw X::Error("expected GPU-projected vocabulary logits");
        return source;
    }
    X::Value Argmax(X::Value source) {
        X::Tensor tensor(source);RequireHost(tensor);auto info=tensor.Info();
        if(info.rank!=2||info.shape[0]<=0||info.shape[1]<=0)throw X::Error("argmax requires nonempty rows");
        auto lease=tensor.Acquire();const auto* data=static_cast<const float*>(info.data);auto result=X::Value::List(Host());
        for(int64_t b=0;b<info.shape[0];++b){int64_t best=0;const auto* row=data+b*info.shape[1];
            for(int64_t d=0;d<info.shape[1];++d){if(!std::isfinite(row[d]))throw X::Error("nonfinite logits");if(row[d]>row[best])best=d;}
            result.Append(X::Value(best));}return result;
    }
    BEGIN_PACKAGE(ModelTensorIO)
        APISET().AddFunc<1>("open",&ModelTensorIO::Open);
        APISET().AddFunc<0>("close",&ModelTensorIO::Close);
        APISET().AddFunc<1>("embedding",&ModelTensorIO::Embedding);
        APISET().AddFunc<1>("zeros",&ModelTensorIO::Zeros);
        APISET().AddFunc<3>("slice",&ModelTensorIO::Slice);
        APISET().AddFunc<2>("read_f32",&ModelTensorIO::ReadF32);
        APISET().AddFunc<2>("visual_token",&ModelTensorIO::VisualToken);
        APISET().AddFunc<1>("logits",&ModelTensorIO::Logits);
        APISET().AddFunc<1>("argmax",&ModelTensorIO::Argmax);
    END_PACKAGE
};
}
