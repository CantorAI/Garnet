// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "xlang3/xlang3.h"
#include "safetensors_index.h"
#include <cstring>
#include <limits>
#include <cmath>
#include <thread>
#include <cstdlib>
#include <algorithm>
#include <fstream>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
namespace Garnet {
// Request-state copies stay native, avoiding millions of script float objects.
class CPUTensorIO {
    SafeTensorsIndex m_index;
    SafeTensorsMappedFile m_mapping;
    const SafeTensorMetadata* m_embedding=nullptr;
    static std::vector<int64_t> Shape(X::Value value) {
        if(!value.IsList())throw X::Error("shape must be a list");
        std::vector<int64_t> shape;
        for(uint64_t i=0;i<value.Size();++i){auto d=value.Get(i).ToInt64();if(d<=0)throw X::Error("shape dimensions must be positive");shape.push_back(d);}
        if(shape.empty())throw X::Error("shape cannot be empty");return shape;
    }
    static size_t Count(const std::vector<int64_t>& shape) {
        size_t count=1;for(auto d:shape){if(size_t(d)>SIZE_MAX/count)throw X::Error("shape overflow");count*=size_t(d);}
        if(count>SIZE_MAX/sizeof(float))throw X::Error("tensor bytes overflow");return count;
    }
public:
    X::Value ReadF32(const std::string& path,X::Value dimensions) {
        auto shape=Shape(dimensions);const auto count=Count(shape);std::vector<float> data(count);
        std::ifstream stream(path,std::ios::binary|std::ios::ate);
        if(!stream||stream.tellg()!=std::streamoff(count*sizeof(float)))throw X::Error("FP32 input file size mismatch");
        stream.seekg(0);stream.read(reinterpret_cast<char*>(data.data()),std::streamsize(count*sizeof(float)));
        if(!stream)throw X::Error("FP32 input file read failed");
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,data.data(),count*sizeof(float));
    }
    X::Value VisualToken(X::Value source,int64_t token) {
        X::Tensor tensor(source);auto info=tensor.Info();
        if(info.symbolic||info.device_type!=0||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=3||!info.data||token<0||token>=info.shape[1])throw X::Error("visual token requires host batch/token/hidden tensor and valid index");
        if(info.strides&&(info.strides[2]!=4||info.strides[1]!=info.shape[2]*4||info.strides[0]!=info.shape[1]*info.shape[2]*4))throw X::Error("visual tokens require contiguous storage");
        auto lease=tensor.Acquire();const auto batch=info.shape[0],hidden=info.shape[2],tokens=info.shape[1];
        std::vector<float> data(size_t(batch)*hidden);auto* input=static_cast<const float*>(info.data);
        for(int64_t b=0;b<batch;++b)std::memcpy(data.data()+b*hidden,input+(b*tokens+token)*hidden,size_t(hidden)*sizeof(float));
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{batch,hidden},data.data(),data.size()*sizeof(float));
    }
    X::Value Zeros(X::Value dimensions) {
        auto shape=Shape(dimensions);std::vector<float> data(Count(shape),0);
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,data.data(),data.size()*sizeof(float));
    }
    X::Value Slice(X::Value source,int64_t start,X::Value dimensions) {
        X::Tensor tensor(source);auto info=tensor.Info();auto shape=Shape(dimensions);auto count=Count(shape);
        if(info.symbolic||info.device_type!=0||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=2||!info.data||start<0||shape[0]!=info.shape[0])throw X::Error("slice requires host FP32 packed rows and matching batch");
        const size_t batch=size_t(shape[0]),width=count/batch;
        if(start>info.shape[1]||width>size_t(info.shape[1]-start))throw X::Error("slice exceeds packed row");
        if(info.strides&& (info.strides[1]!=4||info.strides[0]!=info.shape[1]*4))throw X::Error("slice requires contiguous packed rows");
        auto lease=tensor.Acquire();std::vector<float> data(count);auto* input=static_cast<const float*>(info.data);
        for(size_t b=0;b<batch;++b)std::memcpy(data.data()+b*width,input+b*size_t(info.shape[1])+size_t(start),width*sizeof(float));
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,data.data(),count*sizeof(float));
    }
    bool Open(const std::string& path) {
        m_embedding=nullptr;m_mapping.Close();std::string error;
        if(!m_index.Open(path,error))throw X::Error(error);
        auto* entry=m_index.Find("model.language_model.embed_tokens.weight");
        if(!entry||entry->dataType!="BF16"||entry->shape.size()!=2)throw X::Error("Qwen3.5 BF16 embedding table missing");
        if(entry->shape[0]<=0||entry->shape[1]<=0||uint64_t(entry->shape[0])>UINT64_MAX/2/uint64_t(entry->shape[1])||
            uint64_t(entry->shape[0])*uint64_t(entry->shape[1])*2!=entry->dataSize)throw X::Error("invalid embedding storage size");
        if(!m_mapping.Open(entry->filePath,error))throw X::Error(error);
        if(!m_mapping.DataAt(entry->dataOffset,entry->dataSize))throw X::Error("embedding exceeds mapped checkpoint");
        m_embedding=entry;return true;
    }
    X::Value Embedding(X::Value tokens) {
        if(!m_embedding||!tokens.IsList()||!tokens.Size())throw X::Error("open checkpoint and supply token IDs");
        const auto width=m_embedding->shape[1];std::vector<int64_t> shape{int64_t(tokens.Size()),width};std::vector<float> data(Count(shape));
        auto* bytes=static_cast<const unsigned char*>(m_mapping.DataAt(m_embedding->dataOffset,m_embedding->dataSize));
        for(uint64_t b=0;b<tokens.Size();++b){auto id=tokens.Get(b).ToInt64();if(id<0||id>=m_embedding->shape[0])throw X::Error("embedding token out of range");
            for(int64_t d=0;d<width;++d){uint16_t half;std::memcpy(&half,bytes+2*(id*width+d),2);uint32_t bits=uint32_t(half)<<16;std::memcpy(&data[b*width+d],&bits,4);}}
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,shape,data.data(),data.size()*sizeof(float));
    }
    X::Value Argmax(X::Value source) {
        X::Tensor tensor(source);auto info=tensor.Info();
        if(info.symbolic||info.device_type!=0||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=2||!info.data||info.shape[1]<=0)throw X::Error("argmax requires host FP32 rows");
        if(info.strides&&(info.strides[1]!=4||info.strides[0]!=info.shape[1]*4))throw X::Error("argmax requires contiguous rows");
        auto lease=tensor.Acquire();auto* data=static_cast<const float*>(info.data);auto result=X::Value::List(Host());
        for(int64_t b=0;b<info.shape[0];++b){int64_t best=0;const auto* row=data+b*info.shape[1];
            for(int64_t d=0;d<info.shape[1];++d){if(!std::isfinite(row[d]))throw X::Error("non-finite logits");if(row[d]>row[best])best=d;}result.Append(X::Value(best));}return result;
    }
    X::Value Logits(X::Value source) {
        X::Tensor tensor(source);auto info=tensor.Info();
        if(!m_embedding||info.symbolic||info.device_type!=0||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=2||!info.data||info.shape[1]!=m_embedding->shape[1])throw X::Error("tied head requires normalized host hidden rows");
        if(info.strides&&(info.strides[1]!=4||info.strides[0]!=info.shape[1]*4))throw X::Error("tied head requires contiguous rows");
        auto lease=tensor.Acquire();const int64_t batch=info.shape[0],dim=info.shape[1],vocab=m_embedding->shape[0];
        std::vector<float> output(size_t(batch)*size_t(vocab));auto* hidden=static_cast<const float*>(info.data);
        auto* table=static_cast<const uint16_t*>(m_mapping.DataAt(m_embedding->dataOffset,m_embedding->dataSize));
        int threads=2;
        if(const char* setting=std::getenv("GARNET_OPENVINO_CPU_THREADS"))threads=std::max(1,std::min(64,std::atoi(setting)));
        threads=std::min<int64_t>(threads,vocab);
        auto worker=[&](int id){
            for(int64_t row=vocab*id/threads;row<vocab*(id+1)/threads;++row){
                const auto* weight=table+row*dim;
                for(int64_t b=0;b<batch;++b){const auto* x=hidden+b*dim;int64_t d=0;float sum=0;
#if defined(__aarch64__)
                    float32x4_t accum=vdupq_n_f32(0);
                    float32x4_t accum1=vdupq_n_f32(0),accum2=vdupq_n_f32(0),accum3=vdupq_n_f32(0);
                    // Independent accumulators avoid one long FMA dependency chain.
                    for(;d+16<=dim;d+=16){
                        auto lo=vld1q_u16(weight+d),hi=vld1q_u16(weight+d+8);
                        auto w0=vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(lo)),16));
                        auto w1=vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(lo)),16));
                        auto w2=vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(hi)),16));
                        auto w3=vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(hi)),16));
                        accum=vfmaq_f32(accum,w0,vld1q_f32(x+d));
                        accum1=vfmaq_f32(accum1,w1,vld1q_f32(x+d+4));
                        accum2=vfmaq_f32(accum2,w2,vld1q_f32(x+d+8));
                        accum3=vfmaq_f32(accum3,w3,vld1q_f32(x+d+12));
                    }
                    accum=vaddq_f32(vaddq_f32(accum,accum1),vaddq_f32(accum2,accum3));
                    for(;d+4<=dim;d+=4){auto w=vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vld1_u16(weight+d)),16));accum=vfmaq_f32(accum,w,vld1q_f32(x+d));}
                    sum=vaddvq_f32(accum);
#endif
                    for(;d<dim;++d){uint32_t bits=uint32_t(weight[d])<<16;float w;std::memcpy(&w,&bits,4);sum+=w*x[d];}
                    output[size_t(b)*vocab+row]=sum;
                }
            }
        };
        std::vector<std::thread> workers;
        try{for(int id=1;id<threads;++id)workers.emplace_back(worker,id);worker(0);}
        catch(...){for(auto& thread:workers)thread.join();throw;}
        for(auto& thread:workers)thread.join();
        return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,{batch,vocab},output.data(),output.size()*sizeof(float));
    }
    BEGIN_PACKAGE(CPUTensorIO)
        APISET().AddFunc<1>("zeros",&CPUTensorIO::Zeros);
        APISET().AddFunc<2>("read_f32",&CPUTensorIO::ReadF32);
        APISET().AddFunc<2>("visual_token",&CPUTensorIO::VisualToken);
        APISET().AddFunc<3>("slice",&CPUTensorIO::Slice);
        APISET().AddFunc<1>("open",&CPUTensorIO::Open);
        APISET().AddFunc<1>("embedding",&CPUTensorIO::Embedding);
        APISET().AddFunc<1>("argmax",&CPUTensorIO::Argmax);
        APISET().AddFunc<1>("logits",&CPUTensorIO::Logits);
    END_PACKAGE
};
}
