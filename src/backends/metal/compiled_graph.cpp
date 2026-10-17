// SPDX-License-Identifier: Apache-2.0
#include "compiled_graph.h"
#include "graph_capture.h"
#include "lowering_context.h"
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cmath>
#include <unordered_set>
#include <map>
#include <cstdlib>
#include <cstdio>
namespace Garnet {
size_t MetalCompiledGraph::Allocate(size_t rows,size_t columns) {
    auto id=slots_.size();slots_.push_back({gpu_->Allocate(rows,columns),rows,columns,{int64_t(rows),int64_t(columns)}});return id;
}
size_t MetalCompiledGraph::Weight(const std::string& name) {
    auto found=weights_.find(name);if(found!=weights_.end())return found->second;
    auto entry=index_.Find(name);
    if(!entry)throw std::runtime_error("Metal checkpoint weight missing: "+name);
#ifdef GARNET_METAL_VISION_GLOBAL_ALPHA
    auto ends=[&](const std::string& suffix){return name.size()>=suffix.size()&&name.compare(name.size()-suffix.size(),suffix.size(),suffix)==0;};
    const bool visionProjection=name.rfind("model.visual.blocks.",0)==0&&
        (ends(".attn.qkv.weight")||ends(".attn.proj.weight")||ends(".mlp.linear_fc1.weight")||ends(".mlp.linear_fc2.weight"));
    if(entry->dataType=="BF16"&&entry->shape.size()==2&&visionProjection) {
        const auto id=BF16Weight(name,true);
        if(slots_[id].bf16.GlobalHalfReady())return id;
        // Inexact common scales retain the original FP32-expanded MPS path.
    }
#endif
    if(bf16Projections_&&entry->dataType=="BF16"&&entry->shape.size()==2&&
        (name.rfind("model.language_model.layers.",0)==0||name.rfind("model.layers.",0)==0))return BF16Weight(name);
    const auto modified=std::filesystem::last_write_time(entry->filePath);
    const auto bytes=std::filesystem::file_size(entry->filePath);
    auto shared=sharedWeights_.find(name);
    if(shared!=sharedWeights_.end()) {
        const auto& record=shared->second;const auto& metadata=record.metadata;
        if(metadata.filePath!=entry->filePath||metadata.dataType!=entry->dataType||metadata.shape!=entry->shape||
            metadata.dataOffset!=entry->dataOffset||metadata.dataSize!=entry->dataSize||record.modified!=modified||record.bytes!=bytes)
            throw std::runtime_error("Shared Metal checkpoint changed or does not match");
        auto id=slots_.size();slots_.push_back(record.slot);weights_[name]=id;weightRecords_[name]=record;return id;
    }
    const bool patch=entry&&name=="model.visual.patch_embed.proj.weight"&&entry->shape.size()==5;
    if(!entry||entry->shape.empty()||(!patch&&(entry->shape.size()>3||
        (entry->shape.size()==3&&entry->shape[1]!=1))))
        throw std::runtime_error("Metal matrix weight missing/invalid: "+name);
    for(auto dim:entry->shape)if(dim<=0)throw std::runtime_error("Invalid Metal checkpoint dimension");
    const size_t rows=entry->shape.size()==1?1:entry->shape[0];
    size_t columns=entry->shape.back();
    if(patch){columns=1;for(size_t i=1;i<entry->shape.size();++i){
        if(uint64_t(entry->shape[i])>std::numeric_limits<size_t>::max()/columns)throw std::runtime_error("Metal patch dimension overflow");
        columns*=entry->shape[i];}}
    if(rows>std::numeric_limits<size_t>::max()/columns || rows*columns>std::numeric_limits<size_t>::max()/4)
        throw std::runtime_error("Metal checkpoint matrix size overflow");
    SafeTensorsMappedFile file;std::string error;
    if(!file.Open(entry->filePath,error))throw std::runtime_error(error);
    std::vector<float> data(rows*columns);
    auto raw=file.DataAt(entry->dataOffset,entry->dataSize);
    if(!raw)throw std::runtime_error("Metal checkpoint data is out of file bounds");
    if(entry->dataType=="BF16") {
        if(entry->dataSize!=data.size()*2)throw std::runtime_error("Invalid BF16 weight length");
        for(size_t i=0;i<data.size();++i){uint16_t half;std::memcpy(&half,static_cast<const char*>(raw)+2*i,2);
            uint32_t bits=uint32_t(half)<<16;std::memcpy(&data[i],&bits,4);}
    }else if(entry->dataType=="F32") {
        if(entry->dataSize!=data.size()*4)throw std::runtime_error("Invalid FP32 weight length");
        std::memcpy(data.data(),raw,entry->dataSize);
    }else throw std::runtime_error("Unsupported initial Metal weight dtype: "+entry->dataType);
    const auto id=slots_.size();slots_.push_back({gpu_->Upload(data,rows,columns),rows,columns,
        std::vector<int64_t>(entry->shape.begin(),entry->shape.end())});
    if(modified!=std::filesystem::last_write_time(entry->filePath)||bytes!=std::filesystem::file_size(entry->filePath))
        throw std::runtime_error("Metal checkpoint changed while loading weights");
    weightRecords_[name]={slots_[id],*entry,modified,bytes};
    weights_[name]=id;return id;
}
size_t MetalCompiledGraph::BF16Weight(const std::string& name,bool visionPrefill) {
    const auto key=name+(visionPrefill?"|VisionGlobalBF16":"|BF16");auto found=weights_.find(key);if(found!=weights_.end())return found->second;
    auto entry=index_.Find(name);
    if(!entry||entry->shape.size()!=2||entry->shape[0]<=0||entry->shape[1]<=0)
        throw std::runtime_error("Invalid vocabulary checkpoint matrix");
    // F32 fixtures retain their exact original representation through MPS.
    if(entry->dataType=="F32")return Weight(name);
    if(entry->dataType!="BF16")throw std::runtime_error("BF16 vocabulary head requires BF16/F32 checkpoint weights");
    const auto modified=std::filesystem::last_write_time(entry->filePath);
    const auto bytes=std::filesystem::file_size(entry->filePath);
    auto shared=sharedWeights_.find(key);
    if(shared!=sharedWeights_.end()) {
        const auto& record=shared->second;const auto& metadata=record.metadata;
        if(metadata.filePath!=entry->filePath||metadata.dataType!=entry->dataType||metadata.shape!=entry->shape||
            metadata.dataOffset!=entry->dataOffset||metadata.dataSize!=entry->dataSize||record.modified!=modified||record.bytes!=bytes)
            throw std::runtime_error("Shared BF16 checkpoint changed or does not match");
        const auto id=slots_.size();slots_.push_back(record.slot);weights_[key]=id;weightRecords_[key]=record;return id;
    }
    SafeTensorsMappedFile file;std::string error;
    if(!file.Open(entry->filePath,error))throw std::runtime_error(error);
    auto data=file.DataAt(entry->dataOffset,entry->dataSize);
    if(!data)throw std::runtime_error("Vocabulary checkpoint exceeds file bounds");
    bool floatPrefill=false;
#ifdef GARNET_METAL_FLOAT_DENSE_KEY_PREFILL
    const std::string keySuffix=".self_attn.k_proj.weight";
    floatPrefill=name.rfind("model.layers.",0)==0&&name.size()>=keySuffix.size()&&
        name.compare(name.size()-keySuffix.size(),keySuffix.size(),keySuffix)==0;
#endif
    auto weight=gpu_->UploadBF16(data,entry->dataSize,size_t(entry->shape[0]),size_t(entry->shape[1]),floatPrefill,visionPrefill);
    auto id=slots_.size();slots_.push_back({MetalMatrix(),size_t(entry->shape[0]),size_t(entry->shape[1]),std::vector<int64_t>(entry->shape.begin(),entry->shape.end()),std::move(weight)});
    if(modified!=std::filesystem::last_write_time(entry->filePath)||bytes!=std::filesystem::file_size(entry->filePath))throw std::runtime_error("BF16 checkpoint changed while loading");
    weightRecords_[key]={slots_[id],*entry,modified,bytes};weights_[key]=id;return id;
}
size_t MetalCompiledGraph::LowerVision(size_t input,const X::Value& attrs) {
    auto integer=[&](const char* name,size_t limit){auto v=attrs.Get(name).ToInt64();
        if(v<=0||uint64_t(v)>limit)throw std::runtime_error(std::string("Invalid Metal vision profile: ")+name);return size_t(v);};
    const auto gh=integer("grid_h",128),gw=integer("grid_w",128),hidden=integer("hidden_size",4096),heads=integer("num_heads",128),
        depth=integer("depth",64),patch=integer("patch_size",64),temporal=integer("temporal_patch_size",8),channels=integer("in_channels",16),
        merge=integer("spatial_merge_size",8),positions=integer("num_position_embeddings",65536),outHidden=integer("out_hidden_size",65536);
    const auto theta=attrs.Get("rope_theta").ToDouble();
    const size_t n=gh*gw,unit=merge*merge,feature=channels*temporal*patch*patch,side=size_t(std::sqrt(double(positions)));
    const auto shape=slots_[input].shape;
    if(gh%merge||gw%merge||hidden%heads||(hidden/heads)%4||side*side!=positions||!(theta>0)||!std::isfinite(theta)||
       shape.size()!=3||shape[1]!=int64_t(n)||shape[2]!=int64_t(feature))throw std::runtime_error("Metal still-image profile mismatch");
    const size_t batch=shape[0],dim=hidden/heads;
    auto constant=[&](const std::vector<float>& values,size_t rows,size_t columns){auto id=slots_.size();
        slots_.push_back({gpu_->Upload(values,rows,columns),rows,columns,{int64_t(rows),int64_t(columns)}});return id;};
    auto vision=[&](unsigned mode,size_t a,size_t b,size_t c,size_t rows,size_t columns){
        auto out=Allocate(rows,columns);Op op{Kind::Vision,a,b,out};op.arguments={a,b,c};op.dimensions={mode,n,hidden,heads,unit};ops_.push_back(std::move(op));return out;};
    auto project=[&](size_t a,const std::string& prefix){
        auto w=Weight(prefix+".weight"),bias=Weight(prefix+".bias");
        if(slots_[w].columns!=slots_[a].columns||slots_[bias].rows!=1||slots_[bias].columns!=slots_[w].rows)
            throw std::runtime_error("Metal vision projection checkpoint mismatch: "+prefix);
        auto product=Allocate(slots_[a].rows,slots_[w].rows);ops_.push_back({Kind::MatMul,a,w,product,0,0,true});
        return vision(0,product,bias,bias,slots_[product].rows,slots_[product].columns);
    };
    auto norm=[&](size_t a,const std::string& prefix){auto w=Weight(prefix+".weight"),bias=Weight(prefix+".bias");
        if(slots_[w].rows!=1||slots_[bias].rows!=1||slots_[w].columns!=slots_[a].columns||slots_[bias].columns!=slots_[a].columns)
            throw std::runtime_error("Metal vision norm checkpoint mismatch");
        return vision(1,a,w,bias,slots_[a].rows,slots_[a].columns);};
    auto add=[&](size_t a,size_t b){if(slots_[a].rows!=slots_[b].rows||slots_[a].columns!=slots_[b].columns)
        throw std::runtime_error("Metal vision residual dimensions mismatch");
        auto out=Allocate(slots_[a].rows,slots_[a].columns);ops_.push_back({Kind::Add,a,b,out});return out;};
    // Flatten only input batch/token rows; the copy respects native row padding.
    auto flat=vision(9,input,input,input,batch*n,feature);
    auto patchWeight=Weight("model.visual.patch_embed.proj.weight");
    const std::vector<int64_t> patchShape={int64_t(hidden),int64_t(channels),int64_t(temporal),int64_t(patch),int64_t(patch)};
    if(slots_[patchWeight].shape!=patchShape)throw std::runtime_error("Metal vision patch weight shape mismatch");
    auto x=project(flat,"model.visual.patch_embed.proj");
    if(slots_[x].columns!=hidden)throw std::runtime_error("Metal vision patch hidden size mismatch");
    auto table=Weight("model.visual.pos_embed.weight");
    if(slots_[table].rows!=positions||slots_[table].columns!=hidden)throw std::runtime_error("Metal vision position table mismatch");
    std::vector<float> interpolation,angles;
    interpolation.reserve(n*8);angles.reserve(n*dim);
    for(size_t bh=0;bh<gh;bh+=merge)for(size_t bw=0;bw<gw;bw+=merge)
      for(size_t ih=0;ih<merge;++ih)for(size_t iw=0;iw<merge;++iw){
        const size_t h=bh+ih,w=bw+iw;
        const float ph=gh>1?float(h)*float(side-1)/float(gh-1):0,pw=gw>1?float(w)*float(side-1)/float(gw-1):0;
        const size_t h0=size_t(ph),w0=size_t(pw),h1=std::min(h0+1,side-1),w1=std::min(w0+1,side-1);
        const float dh=ph-h0,dw=pw-w0;
        interpolation.insert(interpolation.end(),{float(h0*side+w0),float(h0*side+w1),float(h1*side+w0),float(h1*side+w1),
            (1-dh)*(1-dw),(1-dh)*dw,dh*(1-dw),dh*dw});
        for(size_t repeat=0;repeat<2;++repeat)for(size_t axis=0;axis<2;++axis)for(size_t d=0;d<dim/4;++d)
            angles.push_back(float(double(axis?w:h)*std::pow(theta,-double(2*d)/double(dim/2))));
      }
    const auto metadata=constant(interpolation,n,8),rope=constant(angles,n,dim);
    x=vision(5,x,table,metadata,batch*n,hidden);
    for(size_t layer=0;layer<depth;++layer){
        auto prefix="model.visual.blocks."+std::to_string(layer);
        auto qkv=project(norm(x,prefix+".norm1"),prefix+".attn.qkv");
        if(slots_[qkv].columns!=3*hidden)throw std::runtime_error("Metal vision QKV shape mismatch");
        auto rotated=vision(6,qkv,rope,rope,batch*n,3*hidden);
        size_t context;
#ifdef GARNET_METAL_FUSED_VISION_CONTEXT
        if(batch*n<=1024&&n<=512&&heads<=16&&dim<=128) {
            context=vision(10,rotated,rotated,rotated,batch*n,hidden);
        } else
#endif
        {
            auto probability=vision(7,rotated,rotated,rotated,batch*n,heads*n);
            context=vision(8,probability,rotated,rotated,batch*n,hidden);
        }
        x=add(x,project(context,prefix+".attn.proj"));
        auto intermediate=project(norm(x,prefix+".norm2"),prefix+".mlp.linear_fc1");
        auto activated=vision(2,intermediate,intermediate,intermediate,slots_[intermediate].rows,slots_[intermediate].columns);
        x=add(x,project(activated,prefix+".mlp.linear_fc2"));
    }
    x=norm(x,"model.visual.merger.norm");
    x=vision(4,x,x,x,batch*n/unit,hidden*unit);
    auto fc=project(x,"model.visual.merger.linear_fc1");
    x=vision(3,fc,fc,fc,slots_[fc].rows,slots_[fc].columns);
    x=project(x,"model.visual.merger.linear_fc2");
    if(slots_[x].columns!=outHidden)throw std::runtime_error("Metal vision merger output width mismatch");
    auto output=vision(4,x,x,x,batch,n/unit*outHidden);
    ops_.back().dimensions[4]=n/unit;
    slots_[output].shape={int64_t(batch),int64_t(n/unit),int64_t(outHidden)};
    return output;
}
void MetalCompiledGraph::Clear() {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    lazyDefaultPlanReady_=false;lazyDefaultSlots_.clear();lazyRetiredBytes_=lazyRehydratedBytes_=0;
#endif
    visionParallelDispatches_=0;
    visionPrefetchDispatches_=0;visionVectorDispatches_=0;mlpGraphDispatches_=0;
    resetKVPrefix_[0]=resetKVPrefix_[1]=false;
    visionGlobalAlphaDispatches_=0;++generation_;ready_=false;bf16Projections_=false;outputParts_.clear();outputStorageGroups_.clear();residentOps_.clear();controlSources_.clear();ops_.clear();inputs_.clear();inputShapes_.clear();tensors_.clear();weights_.clear();sharedWeights_.clear();weightRecords_.clear();bindings_.clear();slots_.clear();gpu_.reset();
}
void MetalCompiledGraph::Close() {std::lock_guard<std::mutex> lock(mutex_);Clear();}
bool MetalCompiledGraph::Build(X::Value graph,X::Value symbols,const std::string& checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);return BuildInternal(graph,symbols,checkpoint,{});
}
bool MetalCompiledGraph::BuildBF16(X::Value graph,X::Value symbols,const std::string& checkpoint) {
    std::lock_guard<std::mutex> lock(mutex_);return BuildInternal(graph,symbols,checkpoint,{},true);
}
bool MetalCompiledGraph::BuildShared(X::Value graph,X::Value symbols,const std::string& checkpoint,X::Value source) {
    if(source.runtime()!=Host()->runtime)throw std::invalid_argument("Shared Metal graph runtime mismatch");
    auto peer=source.NativeData<MetalCompiledGraph>();
    if(!peer||peer==this)throw std::invalid_argument("Shared Metal weights require another compiled graph");
    std::scoped_lock lock(mutex_,peer->mutex_);
    if(!peer->ready_)throw std::invalid_argument("Shared Metal weight source is closed");
    return BuildInternal(graph,symbols,checkpoint,peer->weightRecords_,peer->bf16Projections_);
}
bool MetalCompiledGraph::BuildInternal(X::Value graph,X::Value symbols,const std::string& checkpoint,
    std::unordered_map<std::string,SharedWeight> shared,bool bf16Projections) {
    Clear();sharedWeights_=std::move(shared);bf16Projections_=bf16Projections;
    auto start=std::chrono::steady_clock::now();
    try {
        if(graph.runtime()!=Host()->runtime||symbols.runtime()!=Host()->runtime||!symbols.IsList())
            throw std::runtime_error("Metal graph inputs must belong to this runtime");
        std::string error;if(!index_.Open(checkpoint,error))throw std::runtime_error(error);
        gpu_=std::make_unique<MetalContext>();
        for(uint64_t i=0;i<symbols.Size();++i) {
            X::Tensor symbol(symbols.Get(i));auto info=symbol.Info();
            if(!info.symbolic||info.rank<1||info.rank>4||info.dtype!=X3_TENSOR_FLOAT32)
                throw std::runtime_error("Metal graph input must be symbolic FP32 rank one through four");
            size_t columns=1;
            for(size_t axis=0;axis<info.rank;++axis) {
                if(info.shape[axis]<=0)throw std::runtime_error("Metal graph input dimensions must be positive");
                if(axis) {
                    if(uint64_t(info.shape[axis])>std::numeric_limits<size_t>::max()/columns)
                        throw std::runtime_error("Metal graph input dimensions overflow");
                    columns*=info.shape[axis];
                }
            }
            inputShapes_.emplace_back(info.shape,info.shape+info.rank);
            const auto id=Allocate(info.shape[0],columns);inputs_.push_back(id);slots_[id].shape=inputShapes_.back();
            if(!tensors_.emplace(info.id,id).second)throw std::runtime_error("Duplicate Metal symbolic input");
        }
        TensorGraphCapture capture(graph);
        auto lookup=[&](const X::Value& value){auto found=tensors_.find(TensorId(value));
            if(found==tensors_.end())throw std::runtime_error("Unknown Metal graph operand");
            return found->second;};
        std::map<size_t,std::vector<size_t>> packedParts;
        std::map<size_t,size_t> aliases;
        auto slice=[&](size_t input,size_t groups,size_t sourceWidth,size_t start,size_t count,
            std::vector<int64_t> shape=std::vector<int64_t>{}) {
            if(!groups||!count||start>sourceWidth||count>sourceWidth-start||groups>std::numeric_limits<size_t>::max()/sourceWidth||
                slots_[input].columns!=groups*sourceWidth)throw std::runtime_error("Metal graph slice shape mismatch");
            auto parts=packedParts.find(input);
            if(groups==1&&parts!=packedParts.end()) {
                size_t offset=0;
                for(auto part:parts->second) {
                    if(start==offset&&count==slots_[part].columns) {
                        // A complete packed segment already exists as an independent matrix.
                        // Alias its storage; preserve this view's logical shape separately.
                        auto view=slots_[part];view.shape=shape.empty()?std::vector<int64_t>{int64_t(view.rows),int64_t(view.columns)}:shape;
                        const auto id=slots_.size();slots_.push_back(std::move(view));aliases[id]=part;return id;
                    }
                    offset+=slots_[part].columns;
                }
            }
            auto out=Allocate(slots_[input].rows,groups*count);
            if(!shape.empty())slots_[out].shape=std::move(shape);
            Op op{Kind::Slice,input,input,out};op.dimensions={groups,sourceWidth,start,count};ops_.push_back(std::move(op));return out;
        };
        // Input projections are independent across chunk tokens; recurrent attention stays sequential.
        std::map<size_t,std::pair<size_t,size_t>> chunkOrigins;
        std::map<size_t,size_t> flatChunks;
        std::map<std::pair<size_t,size_t>,size_t> chunkProjections;
        auto matmul=[&](size_t a,size_t b){
            if(slots_[a].columns!=slots_[b].columns)throw std::runtime_error("Metal checkpoint projection shape mismatch");
            auto origin=chunkOrigins.find(a);
            if(origin!=chunkOrigins.end()&&slots_[origin->second.first].shape.size()==3) {
                const auto source=origin->second.first,index=origin->second.second;
                const auto shape=slots_[source].shape;
                const size_t batch=shape[0],chunk=shape[1],width=shape[2],outputWidth=slots_[b].rows;
                auto flat=flatChunks.find(source);
                if(flat==flatChunks.end()) {
                    auto id=Allocate(batch*chunk,width);
                    Op op{Kind::Layout,source,source,id};op.dimensions={0};ops_.push_back(std::move(op));
                    flat=flatChunks.emplace(source,id).first;
                }
                const auto key=std::make_pair(source,b);
                auto projected=chunkProjections.find(key);
                if(projected==chunkProjections.end()) {
                    auto matrix=Allocate(batch*chunk,outputWidth);
                    ops_.push_back({Kind::MatMul,flat->second,b,matrix,0,0,true});
                    auto packed=Allocate(batch,chunk*outputWidth);
                    Op op{Kind::Layout,matrix,matrix,packed};op.dimensions={0};ops_.push_back(std::move(op));
                    projected=chunkProjections.emplace(key,packed).first;
                }
                return slice(projected->second,1,chunk*outputWidth,index*outputWidth,outputWidth);
            }
            auto out=Allocate(slots_[a].rows,slots_[b].rows);ops_.push_back({Kind::MatMul,a,b,out,0,0,true});return out;
        };
        for(const auto& node:capture.Operations()) {
            if(node.provider.empty())continue;
            if(node.provider!="garnet"||node.inputs.empty()||node.operandCount<1||node.operandCount>2)
                throw std::runtime_error("Unsupported initial Metal graph operation: "+node.name);
            auto a=lookup(node.inputs[0]);size_t out;
            if(node.operandCount==2) {
                auto b=lookup(node.inputs[1]);
                if(node.name.rfind("qwen35_conv_bind_",0)==0||node.name.rfind("qwen35_delta_bind_",0)==0||node.name.rfind("qwen35_attention_bind_",0)==0) {
                    static const std::vector<std::string> allowed={"qwen35_conv_bind_weight","qwen35_conv_bind_state",
                        "qwen35_conv_bind_active","qwen35_delta_bind_key","qwen35_delta_bind_value",
                        "qwen35_delta_bind_decay","qwen35_delta_bind_beta","qwen35_delta_bind_state","qwen35_delta_bind_active",
                        "qwen35_attention_bind_key","qwen35_attention_bind_value","qwen35_attention_bind_positions",
                        "qwen35_attention_bind_slots","qwen35_attention_bind_active"};
                    if(std::find(allowed.begin(),allowed.end(),node.name)==allowed.end())
                        throw std::runtime_error("Unknown Metal attention binding: "+node.name);
                    auto bound=bindings_[TensorId(node.inputs[0])];bound[node.name]=b;
                    bindings_[TensorId(node.output)]=std::move(bound);tensors_[TensorId(node.output)]=a;continue;
                }
                if(node.name=="qwen35_concat_sequence") {
                    auto as=slots_[a].shape,bs=slots_[b].shape;
                    if(as.size()!=bs.size()||(as.size()!=2&&as.size()!=3)||as[0]!=bs[0]||(as.size()==3&&as[2]!=bs[2]))
                        throw std::runtime_error("Metal chunk concat shape mismatch");
                    out=Allocate(slots_[a].rows,slots_[a].columns+slots_[b].columns);as[1]+=bs[1];slots_[out].shape=as;
                    Op op{Kind::Concat,a,b,out};op.arguments={a,b,b};op.transpose=false;ops_.push_back(std::move(op));
                    for(auto part:{a,b}) {
                        auto parts=packedParts.find(part);
                        if(parts==packedParts.end())packedParts[out].push_back(part);
                        else packedParts[out].insert(packedParts[out].end(),parts->second.begin(),parts->second.end());
                    }
                    tensors_[TensorId(node.output)]=out;continue;
                }
                if(node.name=="qwen35_mask_chunk") {
                    const auto as=slots_[a].shape,bs=slots_[b].shape;
                    if(as.size()!=3||bs!=std::vector<int64_t>{as[0],as[1]})throw std::runtime_error("Metal chunk mask shape mismatch");
                    out=Allocate(slots_[a].rows,slots_[a].columns);slots_[out].shape=as;
                    Op op{Kind::Layout,a,b,out};op.dimensions={size_t(as[2])};ops_.push_back(std::move(op));
                    tensors_[TensorId(node.output)]=out;continue;
                }
                if(node.name=="qwen35_replace_packed_hidden") {
                    auto hiddenDim=node.attributes.Get("hidden_dim").ToInt64();
                    if(hiddenDim<=0||slots_[a].rows!=slots_[b].rows||slots_[b].columns!=uint64_t(hiddenDim)||
                        slots_[a].columns<=uint64_t(hiddenDim))throw std::runtime_error("Metal packed hidden replacement shape mismatch");
                    const auto packedWidth=slots_[a].columns;
                    auto parts=packedParts.find(a);
                    if(parts!=packedParts.end()&&parts->second.size()==3&&slots_[parts->second[0]].columns==uint64_t(hiddenDim)) {
                        out=Allocate(slots_[a].rows,packedWidth);
                        Op op{Kind::Concat,b,parts->second[1],out};op.arguments={b,parts->second[1],parts->second[2]};op.transpose=true;
                        packedParts[out]=op.arguments;ops_.push_back(std::move(op));
                        tensors_[TensorId(node.output)]=out;continue;
                    }
                    auto tail=slice(a,1,packedWidth,hiddenDim,packedWidth-hiddenDim);
                    out=Allocate(slots_[a].rows,packedWidth);
                    Op op{Kind::Concat,b,tail,out};op.arguments={b,tail,tail};op.transpose=false;ops_.push_back(std::move(op));
                    tensors_[TensorId(node.output)]=out;continue;
                }
                const bool mask=node.name=="qwen35_mask_rows";
                if(node.name!="add"&&!mask)throw std::runtime_error("Metal binary lowering not implemented: "+node.name);
                if(slots_[a].rows!=slots_[b].rows||slots_[b].columns!=(mask?1:slots_[a].columns))
                    throw std::runtime_error("Metal binary shape mismatch");
                out=Allocate(slots_[a].rows,slots_[a].columns);
                ops_.push_back({mask?Kind::Mask:Kind::Add,a,b,out});slots_[out].shape=slots_[a].shape;
                tensors_[TensorId(node.output)]=out;continue;
            }
            if(node.name=="reshape") {
                auto dimensions=node.attributes.Get("shape");std::vector<int64_t> shape;size_t total=slots_[a].rows*slots_[a].columns,known=1;int inferred=-1;
                if(!dimensions.IsList()||!dimensions.Size()||dimensions.Size()>4)throw std::runtime_error("Invalid Metal reshape rank");
                for(uint64_t i=0;i<dimensions.Size();++i){auto d=dimensions.Get(i).ToInt64();shape.push_back(d);
                    if(d==-1){if(inferred!=-1)throw std::runtime_error("Multiple inferred reshape dimensions");inferred=int(i);}
                    else{if(d<=0||uint64_t(d)>total/known)throw std::runtime_error("Invalid Metal reshape dimension");known*=size_t(d);}}
                if(inferred>=0){if(total%known)throw std::runtime_error("Indivisible Metal reshape");shape[inferred]=total/known;}
                else if(known!=total)throw std::runtime_error("Metal reshape element count mismatch");
                out=Allocate(shape[0],total/size_t(shape[0]));slots_[out].shape=shape;
                Op op{Kind::Layout,a,a,out};op.dimensions={0};ops_.push_back(std::move(op));
            }else if(node.name=="qwen35_chunk_token") {
                auto shape=slots_[a].shape;auto index=node.attributes.Get("index").ToInt64();
                if((shape.size()!=2&&shape.size()!=3)||index<0||index>=shape[1])throw std::runtime_error("Invalid Metal chunk token");
                size_t width=shape.size()==3?size_t(shape[2]):1;
                std::vector<int64_t> outputShape=shape.size()==3?std::vector<int64_t>{shape[0],shape[2]}:std::vector<int64_t>{shape[0]};
                out=slice(a,1,slots_[a].columns,size_t(index)*width,width,outputShape);
                chunkOrigins[out]={a,size_t(index)};
                if(std::find(inputs_.begin(),inputs_.end(),a)!=inputs_.end())controlSources_[out]={a,size_t(index)};
            }else if(node.name=="qwen35_vision_encoder") {
                out=LowerVision(a,node.attributes);
            }else if(node.name=="qwen35_attention_step"||node.name=="qwen_text_attention_step"||node.name=="qwen35_attention_chunk"||node.name=="qwen_text_attention_chunk") {
                auto integer=[&](const char* name,size_t limit){auto v=node.attributes.Get(name).ToInt64();
                    if(v<=0||uint64_t(v)>limit)throw std::runtime_error(std::string("Invalid Metal attention dimension: ")+name);
                    return size_t(v);};
                const auto heads=integer("num_heads",128),kv=integer("num_kv_heads",128),dim=integer("head_dim",4096),rotary=integer("rotary_dim",4096);
                const auto theta=node.attributes.Get("rope_theta").ToDouble(),epsilon=node.attributes.Get("eps").ToDouble();
                auto t=node.attributes.Get("mrope_t").ToInt64(),h=node.attributes.Get("mrope_h").ToInt64(),w=node.attributes.Get("mrope_w").ToInt64();
                if(heads%kv||rotary%2||rotary>dim||t<0||h<0||w<0||t>int64_t(rotary/2)||h>int64_t(rotary/2)||w>int64_t(rotary/2)||
                    uint64_t(t+h+w)!=rotary/2||!(theta>0)||!std::isfinite(theta)||!(epsilon>0)||!std::isfinite(float(epsilon)))
                    throw std::runtime_error("Metal attention RoPE configuration invalid");
                const auto bound=bindings_[TensorId(node.inputs[0])];
                auto require=[&](const char* name){auto it=bound.find(name);
                    if(it==bound.end())throw std::runtime_error(std::string("Metal attention missing binding: ")+name);
                    return it->second;};
                auto key=require("qwen35_attention_bind_key"),value=require("qwen35_attention_bind_value"),positions=require("qwen35_attention_bind_positions"),
                    slot=require("qwen35_attention_bind_slots"),active=require("qwen35_attention_bind_active");
                const bool isChunk=node.name=="qwen35_attention_chunk"||node.name=="qwen_text_attention_chunk";
                const auto hs=slots_[a].shape,cs=slots_[key].shape;
                const size_t chunk=isChunk&&hs.size()==3?size_t(hs[1]):1;
                if(hs.size()!=(isChunk?3:2)||cs.size()!=4||cs[0]!=hs[0]||uint64_t(cs[1])!=kv||cs[2]<=0||cs[2]>8192||uint64_t(cs[3])!=dim||
                    slots_[value].shape!=cs||slots_[positions].shape!=(isChunk?std::vector<int64_t>{hs[0],int64_t(chunk),3}:std::vector<int64_t>{hs[0],3})||
                    slots_[slot].shape!=(isChunk?std::vector<int64_t>{hs[0],int64_t(chunk)}:std::vector<int64_t>{hs[0]})||slots_[active].shape!=slots_[slot].shape||
                    (isChunk&&(chunk>kMetalPrefillChunkLimit||cs[2]>2048)))
                    throw std::runtime_error("Metal full attention state shapes invalid");
                // Control tensors must be explicit host inputs for pre-dispatch bounds checks.
                for(auto control:{positions,slot,active})if(std::find(inputs_.begin(),inputs_.end(),control)==inputs_.end()&&!controlSources_.count(control))
                    throw std::runtime_error("Metal attention control tensors must be graph inputs");
                const size_t batch=hs[0],capacity=cs[2],cacheWidth=kv*capacity*dim,rows=batch*chunk,width=hs.back();
                auto flatten=[&](size_t input,size_t columns){auto output=Allocate(rows,columns);Op op{Kind::Layout,input,input,output};op.dimensions={0};ops_.push_back(std::move(op));return output;};
                if(isChunk)a=flatten(a,width);
                const auto ropePositions=isChunk?flatten(positions,3):positions;
                auto prefix=node.attributes.Get("weight_prefix").ToString();
                const bool text=node.name=="qwen_text_attention_step"||node.name=="qwen_text_attention_chunk",normalize=!text||node.attributes.Get("qk_norm").ToInt64()!=0;
                auto qg=matmul(a,Weight(prefix+".q_proj.weight")),kp=matmul(a,Weight(prefix+".k_proj.weight")),vp=matmul(a,Weight(prefix+".v_proj.weight"));
                if(text&&node.attributes.Get("attention_bias").ToInt64()!=0) {
                    auto bias=[&](size_t input,const std::string& name){auto weight=Weight(name);
                        if(slots_[weight].rows!=1||slots_[weight].columns!=slots_[input].columns)throw std::runtime_error("Metal projection bias shape mismatch");
                        auto result=Allocate(slots_[input].rows,slots_[input].columns);ops_.push_back({Kind::Bias,input,weight,result});return result;};
                    qg=bias(qg,prefix+".q_proj.bias");kp=bias(kp,prefix+".k_proj.bias");vp=bias(vp,prefix+".v_proj.bias");
                }
                size_t qw,kw;
                if(normalize){qw=Weight(prefix+".q_norm.weight");kw=Weight(prefix+".k_norm.weight");}
                else{std::vector<float> ones(dim,1);qw=slots_.size();slots_.push_back({gpu_->Upload(ones,1,dim),1,dim,{1,int64_t(dim)}});kw=qw;}
                if(slots_[qg].columns!=(text?1:2)*heads*dim||slots_[kp].columns!=kv*dim||slots_[vp].columns!=kv*dim||
                    slots_[qw].rows!=1||slots_[qw].columns!=dim||slots_[kw].rows!=1||slots_[kw].columns!=dim)
                    throw std::runtime_error("Metal full attention checkpoint dimensions mismatch");
                auto q=text?qg:slice(qg,heads,2*dim,0,dim),gate=text?q:slice(qg,heads,2*dim,dim,dim);
                std::vector<float> frequencies(rotary);
                for(size_t i=0;i<rotary/2;++i) {
                    frequencies[i]=float(std::pow(theta,-2.0*double(i)/double(rotary)));
                    frequencies[rotary/2+i]=(i%3==1&&i<size_t(h)*3)?1.0f:((i%3==2&&i<size_t(w)*3)?2.0f:0.0f);
                }
                auto freq=slots_.size();slots_.push_back({gpu_->Upload(frequencies,2,rotary/2),2,rotary/2,{2,int64_t(rotary/2)}});
                auto qr=Allocate(rows,heads*dim),kr=Allocate(rows,kv*dim);
                Op qrope{Kind::NormRope,q,qw,qr,float(epsilon)};qrope.arguments={q,qw,ropePositions,freq};qrope.dimensions={heads,dim,rotary,size_t(normalize)};qrope.offset=text?0:1;ops_.push_back(std::move(qrope));
                Op krope{Kind::NormRope,kp,kw,kr,float(epsilon)};krope.arguments={kp,kw,ropePositions,freq};krope.dimensions={kv,dim,rotary,size_t(normalize)};krope.offset=text?0:1;ops_.push_back(std::move(krope));
                auto nextKey=Allocate(batch,cacheWidth),nextValue=Allocate(batch,cacheWidth);
                for(auto pair:{std::make_pair(kr,std::make_pair(key,nextKey)),std::make_pair(vp,std::make_pair(value,nextValue))}) {
                    Op update{isChunk?Kind::KVChunk:Kind::KVUpdate,pair.first,pair.second.first,pair.second.second};
                    update.arguments={pair.first,pair.second.first,slot,active,positions};update.dimensions={kv,dim,capacity,chunk};ops_.push_back(std::move(update));
                }
                auto context=Allocate(rows,heads*dim);
                if(isChunk) {
                    Op attend{Kind::AttentionChunk,qr,key,context};attend.arguments={qr,kr,vp,key,value,slot,active,gate};
                    attend.dimensions={heads,kv,dim,capacity,chunk,size_t(!text)};ops_.push_back(std::move(attend));
                } else {
                    auto probabilities=Allocate(batch,heads*capacity);
                    Op scores{Kind::Scores,qr,nextKey,probabilities};scores.arguments={qr,nextKey,slot,active};scores.dimensions={heads,kv,dim,capacity};ops_.push_back(std::move(scores));
                    Op attend{Kind::AttentionContext,probabilities,nextValue,context};attend.arguments={probabilities,nextValue,gate,slot};attend.dimensions={heads,kv,dim,capacity,size_t(!text)};ops_.push_back(std::move(attend));
                }
                auto hidden=matmul(context,Weight(prefix+".o_proj.weight"));
                if(slots_[hidden].columns!=width)throw std::runtime_error("Metal attention output projection shape mismatch");
                if(isChunk) {auto packed=Allocate(batch,chunk*width);Op op{Kind::Layout,hidden,hidden,packed};op.dimensions={0};ops_.push_back(std::move(op));hidden=packed;}
                out=Allocate(batch,chunk*width+2*cacheWidth);
                Op pack{Kind::Concat,hidden,nextKey,out};pack.arguments={hidden,nextKey,nextValue};pack.transpose=true;packedParts[out]=pack.arguments;ops_.push_back(std::move(pack));
            }else if(node.name=="qwen35_attention_output"||node.name=="qwen35_attention_key_cache"||node.name=="qwen35_attention_value_cache") {
                auto dimension=[&](const char* name,size_t limit){auto n=node.attributes.Get(name).ToInt64();
                    if(n<=0||uint64_t(n)>limit)throw std::runtime_error("Invalid Metal attention unpack dimension");
                    return size_t(n);};
                auto hidden=dimension("hidden_dim",kMetalPrefillPackedHiddenLimit),kv=dimension("num_kv_heads",128),dim=dimension("head_dim",4096),capacity=dimension("capacity",8192);
                const size_t cache=kv*dim*capacity,width=hidden+2*cache;const auto batch=int64_t(slots_[a].rows);
                if(slots_[a].columns!=width)throw std::runtime_error("Metal attention packed shape mismatch");
                if(node.name=="qwen35_attention_output")out=slice(a,1,width,0,hidden,{batch,int64_t(hidden)});
                else out=slice(a,1,width,hidden+(node.name=="qwen35_attention_value_cache"?cache:0),cache,
                    {batch,int64_t(kv),int64_t(capacity),int64_t(dim)});
            }else if(node.name=="qwen35_linear_step"||node.name=="qwen35_linear_chunk") {
                auto integer=[&](const char* name,size_t limit){auto v=node.attributes.Get(name).ToInt64();
                    if(v<=0||uint64_t(v)>limit)throw std::runtime_error(std::string("Invalid Metal linear dimension: ")+name);
                    return size_t(v);};
                const auto kh=integer("key_heads",128),vh=integer("value_heads",128),kd=integer("key_dim",4096),vd=integer("value_dim",4096);
                if(vh%kh)throw std::runtime_error("Metal linear head grouping mismatch");
                const auto hiddenShape=slots_[a].shape;
                const bool isChunk=node.name=="qwen35_linear_chunk";
                const size_t chunk=isChunk&&hiddenShape.size()==3?size_t(hiddenShape[1]):1;
                const auto bound=bindings_[TensorId(node.inputs[0])];
                auto require=[&](const char* name){auto it=bound.find(name);
                    if(it==bound.end())throw std::runtime_error(std::string("Metal linear missing binding: ")+name);
                    return it->second;};
                auto convState=require("qwen35_conv_bind_state"),state=require("qwen35_delta_bind_state"),active=require("qwen35_conv_bind_active");
                const auto cs=slots_[convState].shape;
                const size_t channels=2*kh*kd+vh*vd;
                if(hiddenShape.size()!=(isChunk?3:2)||chunk>kMetalPrefillChunkLimit||cs.size()!=3||cs[0]!=hiddenShape[0]||uint64_t(cs[1])!=channels||cs[2]<=0||cs[2]>64||
                    slots_[state].shape!=std::vector<int64_t>{hiddenShape[0],int64_t(vh),int64_t(kd),int64_t(vd)}||
                    slots_[active].shape!=(isChunk?std::vector<int64_t>{hiddenShape[0],int64_t(chunk)}:std::vector<int64_t>{hiddenShape[0]}))throw std::runtime_error("Metal linear state logical shape mismatch");
                const auto batch=size_t(hiddenShape[0]),kernel=size_t(cs[2]);
                const size_t width=size_t(hiddenShape.back());
                auto layout=[&](size_t source,size_t rows,size_t columns) {
                    auto result=Allocate(rows,columns);Op op{Kind::Layout,source,source,result};op.dimensions={0};ops_.push_back(std::move(op));return result;
                };
                auto source=isChunk?layout(a,batch*chunk,width):a;
                const auto prefix=node.attributes.Get("weight_prefix").ToString();
                const auto epsilon=float(node.attributes.Get("eps").ToDouble());
                if(!(epsilon>0)||!std::isfinite(epsilon))throw std::runtime_error("Invalid Metal linear epsilon");
                auto qkv=matmul(source,Weight(prefix+".in_proj_qkv.weight"));
                auto z=matmul(source,Weight(prefix+".in_proj_z.weight"));
                auto ap=matmul(source,Weight(prefix+".in_proj_a.weight")),bp=matmul(source,Weight(prefix+".in_proj_b.weight"));
                auto convWeight=Weight(prefix+".conv1d.weight"),aLog=Weight(prefix+".A_log"),dt=Weight(prefix+".dt_bias");
                if(slots_[qkv].columns!=channels||slots_[z].columns!=vh*vd||slots_[ap].columns!=vh||slots_[bp].columns!=vh||
                    slots_[convWeight].rows!=channels||slots_[convWeight].columns!=kernel||slots_[aLog].columns!=vh||slots_[dt].columns!=vh)
                    throw std::runtime_error("Metal linear checkpoint projection shape mismatch");
                // Projections share weights across all tokens. State updates
                // retain causal order and honor every row's active mask.
                auto qkvTokens=isChunk?layout(qkv,batch,chunk*channels):qkv;
                auto zTokens=isChunk?layout(z,batch,chunk*vh*vd):z;
                auto aTokens=isChunk?layout(ap,batch,chunk*vh):ap;
                auto bTokens=isChunk?layout(bp,batch,chunk*vh):bp;
                auto norm=Weight(prefix+".norm.weight");
                if(slots_[norm].rows!=1||slots_[norm].columns!=vd)throw std::runtime_error("Metal linear norm weight shape mismatch");
                const size_t qw=vh*kd,vw=vh*vd,pw=2*qw+vw+2*vh;
                bool chunkConv=false;size_t preparedTokens=0,preparedFlat=0;
#ifdef GARNET_METAL_LINEAR_CHUNK_CONV
                chunkConv=isChunk&&kernel<=8;
                if(chunkConv) {
                    auto convolution=Allocate(batch*chunk,channels),nextConv=Allocate(batch,channels*kernel);
                    Op conv{Kind::ConvChunk,qkv,convWeight,convolution};conv.arguments={qkv,convWeight,convState,active};
                    conv.dimensions={chunk,kernel};conv.auxiliaryOutput=nextConv;ops_.push_back(std::move(conv));convState=nextConv;
                    auto prepared=Allocate(batch*chunk,pw);
                    preparedFlat=prepared;
                    Op prepare{Kind::Prepare,convolution,ap,prepared};prepare.arguments={convolution,ap,bp,aLog,dt};
                    prepare.dimensions={kh,vh,kd,vd,0};ops_.push_back(std::move(prepare));
                    preparedTokens=layout(prepared,batch,chunk*pw);
                }
#endif
                bool chunkDelta=false;size_t gatedFlat=0;
#ifdef GARNET_METAL_LINEAR_CHUNK_DELTA
                chunkDelta=chunkConv&&kd==128&&vd%16==0;
                if(chunkDelta) {
                    auto cores=Allocate(batch*chunk,vw),nextState=Allocate(batch,vh*kd*vd);
                    Op delta{Kind::DeltaChunk,preparedFlat,state,cores};delta.arguments={preparedFlat,state,active};
                    delta.dimensions={chunk,vh,kd,vd};delta.auxiliaryOutput=nextState;ops_.push_back(std::move(delta));state=nextState;
                    gatedFlat=Allocate(batch*chunk,vw);
                    Op gate{Kind::GateNorm,cores,z,gatedFlat,epsilon};gate.arguments={cores,z,norm};gate.dimensions={vh,vd};ops_.push_back(std::move(gate));
                }
#endif
                size_t accumulated=0;
                bool directWrite=false;
#ifdef GARNET_METAL_LINEAR_CHUNK_DIRECT_WRITE
                directWrite=isChunk;
                if(directWrite&&!chunkDelta)accumulated=Allocate(batch,chunk*vw);
#endif
                for(size_t token=0;token<chunk&&!chunkDelta;++token) {
                    auto select=[&](size_t projected,size_t columns) {
                        return isChunk?slice(projected,1,chunk*columns,token*columns,columns):projected;
                    };
                    auto tokenActive=isChunk?slice(active,1,chunk,token,1,{int64_t(batch)}):active;
                    auto tokenZ=select(zTokens,vw);size_t convPacked=0,prepared=0;
                    if(chunkConv)prepared=select(preparedTokens,pw);
                    else {
                        auto tokenQkv=select(qkvTokens,channels),tokenA=select(aTokens,vh),tokenB=select(bTokens,vh);
                        convPacked=Allocate(batch,channels*(kernel+1));
                        Op conv{Kind::Conv,tokenQkv,convWeight,convPacked};conv.arguments={tokenQkv,convWeight,convState,tokenActive};conv.dimensions={kernel};ops_.push_back(std::move(conv));
                        prepared=Allocate(batch,pw);
                        Op prepare{Kind::Prepare,convPacked,tokenA,prepared};prepare.arguments={convPacked,tokenA,tokenB,aLog,dt};
                        prepare.dimensions={kh,vh,kd,vd,kernel};ops_.push_back(std::move(prepare));
                    }
                    auto q=slice(prepared,1,pw,0,qw),k=slice(prepared,1,pw,qw,qw),v=slice(prepared,1,pw,2*qw,vw),
                        g=slice(prepared,1,pw,2*qw+vw,vh),beta=slice(prepared,1,pw,2*qw+vw+vh,vh);
                    auto deltaPacked=Allocate(batch,vw*(kd+1));
                    Op delta{Kind::Delta,q,k,deltaPacked};delta.arguments={q,k,v,g,beta,state,tokenActive};delta.dimensions={vh,kd,vd};ops_.push_back(std::move(delta));
                    auto core=slice(deltaPacked,vh,(kd+1)*vd,0,vd);
                    auto gated=directWrite?accumulated:Allocate(batch,vw);
                    Op gate{Kind::GateNorm,core,tokenZ,gated,epsilon};gate.arguments={core,tokenZ,norm};gate.dimensions={vh,vd};ops_.push_back(std::move(gate));
                    if(directWrite)ops_.back().dimensions.push_back(token*vw);
                    else if(token==0)accumulated=gated;
                    else {
                        auto joined=Allocate(batch,(token+1)*vw);Op join{Kind::Concat,accumulated,gated,joined};
                        join.arguments={accumulated,gated,gated};join.transpose=false;ops_.push_back(std::move(join));accumulated=joined;
                    }
                    if(!chunkConv)convState=slice(convPacked,channels,kernel+1,1,kernel);
                    state=slice(deltaPacked,vh,(kd+1)*vd,vd,kd*vd);
                }
                auto gated=chunkDelta?gatedFlat:(isChunk?layout(accumulated,batch*chunk,vw):accumulated);
                auto hidden=matmul(gated,Weight(prefix+".out_proj.weight"));
                if(slots_[hidden].columns!=width)throw std::runtime_error("Metal linear output projection shape mismatch");
                if(isChunk)hidden=layout(hidden,batch,chunk*width);
                out=Allocate(batch,slots_[hidden].columns+slots_[convState].columns+slots_[state].columns);
                Op pack{Kind::Concat,hidden,convState,out};pack.arguments={hidden,convState,state};pack.transpose=true;packedParts[out]=pack.arguments;ops_.push_back(std::move(pack));
            }else if(node.name=="qwen35_linear_output"||node.name=="qwen35_linear_conv_state"||node.name=="qwen35_linear_recurrent_state") {
                auto dim=[&](const char* name,size_t limit){auto v=node.attributes.Get(name).ToInt64();
                    if(v<=0||uint64_t(v)>limit)throw std::runtime_error("Invalid Metal linear unpack dimension");
                    return size_t(v);};
                const auto h=dim("hidden_dim",kMetalPrefillPackedHiddenLimit),kh=dim("key_heads",128),vh=dim("value_heads",128),kd=dim("key_dim",4096),
                    vd=dim("value_dim",4096),kernel=dim("kernel_size",64),channels=2*kh*kd+vh*vd,conv=channels*kernel,state=vh*kd*vd;
                if(vh%kh||slots_[a].columns!=h+conv+state)throw std::runtime_error("Metal linear packed dimensions mismatch");
                const auto batch=int64_t(slots_[a].rows);
                if(node.name=="qwen35_linear_output")out=slice(a,1,h+conv+state,0,h,{batch,int64_t(h)});
                else if(node.name=="qwen35_linear_conv_state")out=slice(a,1,h+conv+state,h,conv,{batch,int64_t(channels),int64_t(kernel)});
                else out=slice(a,1,h+conv+state,h+conv,state,{batch,int64_t(vh),int64_t(kd),int64_t(vd)});
            }else if(node.name=="qwen35_conv_output"||node.name=="qwen35_conv_state"||
                node.name=="qwen35_delta_output"||node.name=="qwen35_delta_state") {
                const auto shape=slots_[a].shape;
                if(shape.size()!=3)throw std::runtime_error("Metal attention unpack requires rank three packed input");
                const bool delta=node.name.rfind("qwen35_delta_",0)==0,wantState=node.name.find("_state")!=std::string::npos;
                size_t start=0,count=1;std::vector<int64_t> outShape;
                if(delta) {
                    const auto kd=node.attributes.Get("key_dim").ToInt64(),vd=node.attributes.Get("value_dim").ToInt64();
                    if(kd<=0||vd<=0||uint64_t(kd)>=std::numeric_limits<size_t>::max()/uint64_t(vd)||
                        uint64_t(shape[2])!=(uint64_t(kd)+1)*uint64_t(vd))throw std::runtime_error("Metal delta unpack dimensions invalid");
                    start=wantState?vd:0;count=wantState?size_t(kd)*size_t(vd):size_t(vd);
                    outShape=wantState?std::vector<int64_t>{shape[0],shape[1],kd,vd}:std::vector<int64_t>{shape[0],shape[1],vd};
                }else {
                    const auto kernel=node.attributes.Get("kernel_size").ToInt64();
                    if(kernel<=0||uint64_t(kernel)==std::numeric_limits<size_t>::max()||uint64_t(shape[2])!=uint64_t(kernel)+1)
                        throw std::runtime_error("Metal convolution unpack dimensions invalid");
                    start=wantState?1:0;count=wantState?kernel:1;
                    outShape=wantState?std::vector<int64_t>{shape[0],shape[1],kernel}:std::vector<int64_t>{shape[0],shape[1]};
                }
                out=Allocate(slots_[a].rows,size_t(shape[1])*count);slots_[out].shape=std::move(outShape);
                Op op{Kind::Slice,a,a,out};op.dimensions={size_t(shape[1]),size_t(shape[2]),start,count};ops_.push_back(std::move(op));
            }else if(node.name=="qwen35_conv_step"||node.name=="qwen35_delta_step") {
                const auto bound=bindings_[TensorId(node.inputs[0])];
                auto require=[&](const std::string& name){auto it=bound.find(name);
                    if(it==bound.end())throw std::runtime_error("Metal attention missing binding: "+name);
                    return it->second;};
                const auto shape=slots_[a].shape;
                if(node.name=="qwen35_conv_step") {
                    auto w=require("qwen35_conv_bind_weight"),s=require("qwen35_conv_bind_state"),active=require("qwen35_conv_bind_active");
                    const auto ss=slots_[s].shape;
                    if(shape.size()!=2||ss.size()!=3||ss[0]!=shape[0]||ss[1]!=shape[1]||
                        slots_[w].shape!=std::vector<int64_t>{shape[1],ss[2]}||slots_[active].shape!=std::vector<int64_t>{shape[0]})
                        throw std::runtime_error("Metal causal convolution logical shape mismatch");
                    const size_t kernel=ss[2];
                    if(slots_[s].columns>std::numeric_limits<size_t>::max()-slots_[a].columns)
                        throw std::runtime_error("Metal convolution packed shape overflow");
                    out=Allocate(slots_[a].rows,slots_[a].columns+slots_[s].columns);
                    slots_[out].shape={shape[0],shape[1],int64_t(kernel+1)};
                    Op op{Kind::Conv,a,w,out};op.arguments={a,w,s,active};op.dimensions={kernel};ops_.push_back(std::move(op));
                }else {
                    auto k=require("qwen35_delta_bind_key"),v=require("qwen35_delta_bind_value"),g=require("qwen35_delta_bind_decay"),
                        beta=require("qwen35_delta_bind_beta"),s=require("qwen35_delta_bind_state"),active=require("qwen35_delta_bind_active");
                    const auto vs=slots_[v].shape;
                    if(shape.size()!=3||vs.size()!=3||vs[0]!=shape[0]||vs[1]!=shape[1]||slots_[k].shape!=shape||
                        slots_[s].shape!=std::vector<int64_t>{shape[0],shape[1],shape[2],vs[2]}||
                        slots_[g].shape!=std::vector<int64_t>{shape[0],shape[1]}||slots_[beta].shape!=slots_[g].shape||
                        slots_[active].shape!=std::vector<int64_t>{shape[0]})throw std::runtime_error("Metal gated delta logical shape mismatch");
                    if(slots_[s].columns>std::numeric_limits<size_t>::max()-slots_[v].columns)
                        throw std::runtime_error("Metal delta packed shape overflow");
                    out=Allocate(slots_[a].rows,slots_[v].columns+slots_[s].columns);
                    slots_[out].shape={shape[0],shape[1],int64_t((size_t(shape[2])+1)*size_t(vs[2]))};
                    Op op{Kind::Delta,a,k,out};op.arguments={a,k,v,g,beta,s,active};
                    op.dimensions={size_t(shape[1]),size_t(shape[2]),size_t(vs[2])};ops_.push_back(std::move(op));
                }
            }else if(node.name=="qwen35_rms_norm"||node.name=="qwen_text_norm") {
                auto weight=Weight(node.attributes.Get("weight_name").ToString());
                if(slots_[weight].rows!=1||slots_[weight].columns!=slots_[a].columns)
                    throw std::runtime_error("Metal normalization shape mismatch");
                out=Allocate(slots_[a].rows,slots_[a].columns);
                ops_.push_back({Kind::Norm,a,weight,out,float(node.attributes.Get("eps").ToDouble()),node.name=="qwen_text_norm"?0.0f:1.0f,false});
            }else if(node.name=="qwen35_mlp") {
                auto prefix=node.attributes.Get("weight_prefix").ToString();
                auto gate=matmul(a,Weight(prefix+".gate_proj.weight"));
                auto up=matmul(a,Weight(prefix+".up_proj.weight"));
                if(slots_[gate].columns!=slots_[up].columns)throw std::runtime_error("Metal gate/up shape mismatch");
                auto activated=Allocate(slots_[gate].rows,slots_[gate].columns);
                ops_.push_back({Kind::Silu,gate,up,activated});
                out=matmul(activated,Weight(prefix+".down_proj.weight"));
            }else if(node.name=="qwen35_tied_head"||node.name=="qwen_text_head") {
                const auto name=node.name=="qwen35_tied_head"?"model.language_model.embed_tokens.weight":node.attributes.Get("weight_name").ToString();
                const bool bf16=node.attributes.Get("bf16_weight").ToInt64()!=0;
                auto weight=bf16?BF16Weight(name):Weight(name);
#ifdef GARNET_METAL_SCREENED_HEAD
                if(bf16&&slots_[weight].bf16.Valid())gpu_->PrepareScreenedHead(slots_[weight].bf16);
#endif
                if(bf16&&index_.Find(name)->dataType=="BF16") {
                    if(slots_[a].columns!=slots_[weight].columns||slots_[a].rows>2)throw std::runtime_error("BF16 vocabulary profile requires batch 1/2");
                    out=Allocate(slots_[a].rows,slots_[weight].rows);ops_.push_back({Kind::BF16MatVec,a,weight,out});
                }else out=matmul(a,weight);
            }
            else throw std::runtime_error("Metal lowering not implemented: "+node.name);
            tensors_[TensorId(node.output)]=out;
        }
        auto output=capture.Outputs();
        if((output.IsList()||x3_value_object_kind(output.raw())==X3_OBJECT_KIND_TUPLE)&&output.Size()==1)
            output=output.Get(uint64_t(0));
        if(!IsTensor(output))throw std::runtime_error("Initial Metal graph requires one tensor output");
        output_=lookup(output);
        if(packedParts.count(output_))outputParts_=packedParts.at(output_);
        // All operations write fresh slots. Trace only matrices needed by the
        // returned value, resolving storage aliases before removing unused packs.
        auto canonical=[&](size_t id){while(aliases.count(id))id=aliases.at(id);return id;};
        // A delta temporary consumed only by its complete output/state slices
        // can write those independent matrices directly, without packing/copying.
        std::unordered_set<size_t> removedSlices;
        for(size_t index=0;index<ops_.size();++index) {
            auto& delta=ops_[index];if(delta.kind!=Kind::Delta)continue;
            const auto packed=canonical(delta.out),heads=delta.dimensions[0],kd=delta.dimensions[1],vd=delta.dimensions[2];
            if(packed==canonical(output_))continue;
            size_t core=SIZE_MAX,state=SIZE_MAX;bool eligible=true;
            for(size_t consumer=0;consumer<ops_.size();++consumer) {
                if(consumer==index)continue;
                const auto& op=ops_[consumer];
                bool uses=canonical(op.a)==packed||canonical(op.b)==packed;
                for(auto arg:op.arguments)uses=uses||canonical(arg)==packed;
                if(!uses)continue;
                if(op.kind!=Kind::Slice||consumer<=index||op.dimensions.size()!=4||
                    op.dimensions[0]!=heads||op.dimensions[1]!=(kd+1)*vd){eligible=false;break;}
                if(op.dimensions[2]==0&&op.dimensions[3]==vd&&core==SIZE_MAX)core=consumer;
                else if(op.dimensions[2]==vd&&op.dimensions[3]==kd*vd&&state==SIZE_MAX)state=consumer;
                else {eligible=false;break;}
            }
            if(eligible&&core!=SIZE_MAX&&state!=SIZE_MAX) {
                delta.kind=Kind::DeltaSplit;delta.out=ops_[core].out;delta.auxiliaryOutput=ops_[state].out;
                removedSlices.insert(core);removedSlices.insert(state);
            }
        }
        if(!removedSlices.empty()) {
            std::vector<Op> fused;
            for(size_t index=0;index<ops_.size();++index)if(!removedSlices.count(index))fused.push_back(std::move(ops_[index]));
            ops_=std::move(fused);
        }
        std::unordered_set<size_t> live={canonical(output_)};
        std::vector<Op> retained;
        for(auto it=ops_.rbegin();it!=ops_.rend();++it) {
            if(!live.count(canonical(it->out))&&(it->auxiliaryOutput==SIZE_MAX||!live.count(canonical(it->auxiliaryOutput))))continue;
            live.insert(canonical(it->out));if(it->auxiliaryOutput!=SIZE_MAX)live.insert(canonical(it->auxiliaryOutput));
            live.insert(canonical(it->a));live.insert(canonical(it->b));
            for(auto arg:it->arguments)live.insert(canonical(arg));
            retained.push_back(std::move(*it));
        }
        std::reverse(retained.begin(),retained.end());ops_=std::move(retained);
        if(outputParts_.size()==3) {
            std::unordered_set<size_t> needed;
            for(auto part:outputParts_)needed.insert(canonical(part));
            residentOps_.assign(ops_.size(),0);
            for(size_t i=ops_.size();i-->0;) {
                auto& op=ops_[i];if(!needed.count(canonical(op.out))&&(op.auxiliaryOutput==SIZE_MAX||!needed.count(canonical(op.auxiliaryOutput))))continue;
                residentOps_[i]=1;needed.insert(canonical(op.a));needed.insert(canonical(op.b));
                for(auto argument:op.arguments)needed.insert(canonical(argument));
            }
        }
        for(auto id:inputs_)live.insert(canonical(id));
        for(size_t id=0;id<slots_.size();++id)if(!live.count(canonical(id)))slots_[id].matrix=MetalMatrix();
        // Exchange only fresh, independently written output storage. Keep
        // aliases together and retain the copy path for inputs/constants or
        // storage shared by two returned parts.
        outputStorageGroups_.resize(outputParts_.size());
        for(size_t part=0;part<outputParts_.size();++part) {
            const auto& matrix=slots_[outputParts_[part]].matrix;std::vector<size_t> group;
            bool eligible=true,writer=false;
            for(size_t id=0;id<slots_.size();++id)if(matrix.SharesStorage(slots_[id].matrix)) {
                group.push_back(id);
                if(!matrix.CompatibleLayout(slots_[id].matrix)||std::find(inputs_.begin(),inputs_.end(),id)!=inputs_.end())eligible=false;
            }
            for(size_t other=0;other<outputParts_.size();++other)if(other!=part&&matrix.SharesStorage(slots_[outputParts_[other]].matrix))eligible=false;
            for(size_t i=0;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]&&
                (std::find(group.begin(),group.end(),ops_[i].out)!=group.end()||std::find(group.begin(),group.end(),ops_[i].auxiliaryOutput)!=group.end()))writer=true;
            if(eligible&&writer)outputStorageGroups_[part]=std::move(group);
        }
        resetKVPrefix_[0]=ComputeResetKVPrefix(1);resetKVPrefix_[1]=ComputeResetKVPrefix(2);
#ifdef GARNET_METAL_BODY_SHAPE
        // Prepare before readiness/compile timing ends. Decode never lazily compiles.
        for(const auto& op:ops_)if((op.kind==Kind::BF16MatVec||(op.kind==Kind::MatMul&&op.transpose))&&slots_[op.b].bf16.Valid())
            gpu_->PrepareBF16BodyShape(slots_[op.a].matrix,slots_[op.b].bf16,slots_[op.out].matrix);
#endif
        sharedWeights_.clear();ready_=true;
        compileMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        return true;
    }catch(...){Clear();throw;}
}
bool MetalCompiledGraph::CanUpdateKVInPlace(size_t part) const {
#ifdef GARNET_METAL_RESIDENT_KV_INPLACE
    if(part<1||part>2||inputs_.size()!=6||inputShapes_[0].size()!=2||
        part>=outputStorageGroups_.size()||outputStorageGroups_[part].empty())return false;
    const auto& group=outputStorageGroups_[part];size_t writer=SIZE_MAX;
    auto member=[&](size_t id){return std::find(group.begin(),group.end(),id)!=group.end();};
    for(size_t i=0;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]&&
        (member(ops_[i].out)||member(ops_[i].auxiliaryOutput))) {
        const auto& op=ops_[i];
        if(writer!=SIZE_MAX||op.kind!=Kind::KVUpdate||op.arguments.size()!=5||op.arguments[1]!=inputs_[part])return false;
        writer=i;
    }
    if(writer==SIZE_MAX)return false;
    // Later operations must consume the updated cache, not its old input view.
    const auto& source=slots_[inputs_[part]].matrix;
    for(size_t i=writer+1;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]) {
        const auto& op=ops_[i];
        if(op.a==inputs_[part]||op.b==inputs_[part]||source.SharesStorage(slots_[op.a].matrix)||source.SharesStorage(slots_[op.b].matrix))return false;
        for(auto id:op.arguments)if(id==inputs_[part]||source.SharesStorage(slots_[id].matrix))return false;
    }
    return true;
#else
    return false;
#endif
}
bool MetalCompiledGraph::ComputeResetKVPrefix(size_t part) const {
    if(part<1||part>2||inputs_.size()!=6||part>=outputStorageGroups_.size()||outputStorageGroups_[part].empty())return false;
    const auto& shape=inputShapes_[part];
    if(shape.size()!=4||std::any_of(shape.begin(),shape.end(),[](int64_t n){return n<=0;}))return false;
    const auto& group=outputStorageGroups_[part];
    auto member=[&](size_t id){return std::find(group.begin(),group.end(),id)!=group.end();};
    auto target=slots_[outputParts_[part]].matrix;std::unordered_set<size_t> visited;
    for(;;) {
        size_t writer=SIZE_MAX;
        auto writes=[&](size_t id){return id!=SIZE_MAX&&target.SharesStorage(slots_[id].matrix);};
        for(size_t i=0;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]&&
            (writes(ops_[i].out)||writes(ops_[i].auxiliaryOutput))) {
            if(writer!=SIZE_MAX)return false;writer=i;
        }
        if(writer==SIZE_MAX||!visited.insert(writer).second)return false;
        const auto& op=ops_[writer];
        if(op.kind==Kind::Layout) {
            // Prefill returns the cache through a fresh reshape copy. Admit
            // only a complete, row-preserving copy, then prove its producer.
            const auto& input=slots_[op.a];const auto& output=slots_[op.out];
            if(op.a!=op.b||op.dimensions!=std::vector<size_t>{0}||op.auxiliaryOutput!=SIZE_MAX||
                input.rows!=size_t(shape[0])||output.rows!=input.rows||input.columns!=slots_[inputs_[part]].columns||output.columns!=input.columns)
                return false;
            target=input.matrix;continue;
        }
        if((op.kind!=Kind::KVUpdate&&op.kind!=Kind::KVChunk)||op.arguments.size()!=5||
            op.arguments[1]!=inputs_[part]||op.arguments[2]!=inputs_[4]||op.arguments[3]!=inputs_[5]||op.dimensions.size()!=4||
            op.dimensions[0]!=size_t(shape[1])||op.dimensions[1]!=size_t(shape[3])||op.dimensions[2]!=size_t(shape[2]))return false;
        break;
    }
    // Reject a writer aliasing the input cache outside the proved output group.
    const auto& source=slots_[inputs_[part]].matrix;
    for(size_t i=0;i<ops_.size();++i)if(i<residentOps_.size()&&residentOps_[i]) {
        const auto& op=ops_[i];
        if(!member(op.out)&&source.SharesStorage(slots_[op.out].matrix))return false;
        if(op.auxiliaryOutput!=SIZE_MAX&&!member(op.auxiliaryOutput)&&source.SharesStorage(slots_[op.auxiliaryOutput].matrix))return false;
    }
    return true;
}
void MetalCompiledGraph::PrepareInputs(X::Value inputs,const std::unordered_set<size_t>& skip) {
    if(!ready_||inputs.runtime()!=Host()->runtime||!inputs.IsList()||inputs.Size()!=inputs_.size())
        throw std::runtime_error("Metal graph not ready or input count invalid");
    std::unordered_map<size_t,std::vector<float>> hostInputs;
    std::unordered_set<size_t> controls;
    for(const auto& op:ops_)if(op.kind==Kind::KVUpdate||op.kind==Kind::KVChunk)
        for(auto i:{2,3,4}){auto id=op.arguments[i];controls.insert(controlSources_.count(id)?controlSources_.at(id).first:id);}
    for(const auto& op:ops_)if(op.kind==Kind::Layout&&op.dimensions[0])controls.insert(op.b);
    for(size_t i=0;i<inputs_.size();++i) {
        if(skip.count(i))continue;
        X::Tensor tensor(inputs.Get(i));auto info=tensor.Info();auto& slot=slots_[inputs_[i]];
        if(info.symbolic||info.device_type!=0||info.dtype!=X3_TENSOR_FLOAT32||info.rank!=inputShapes_[i].size()||
            !std::equal(inputShapes_[i].begin(),inputShapes_[i].end(),info.shape)||!info.data||!IsContiguousTensor(tensor))
            throw std::runtime_error("Metal runtime input shape/storage mismatch");
        auto lease=tensor.Acquire();auto data=static_cast<const float*>(info.data);
        if(controls.count(inputs_[i]))hostInputs[inputs_[i]]=std::vector<float>(data,data+slot.rows*slot.columns);
    }
    for(const auto& op:ops_)if(op.kind==Kind::Layout&&op.dimensions[0]) {
        const auto& flags=hostInputs.at(op.b);
        for(float value:flags)if(!std::isfinite(value)||(value!=0&&value!=1))throw std::runtime_error("Invalid Metal chunk active flags");
    }
    for(const auto& entry:controlSources_)if(controls.count(entry.second.first)) {
        const auto& source=slots_[entry.second.first];const auto& values=hostInputs.at(entry.second.first);
        const auto& output=slots_[entry.first];std::vector<float> selected(output.rows*output.columns);
        for(size_t row=0;row<output.rows;++row)std::copy_n(values.data()+row*source.columns+entry.second.second*output.columns,
            output.columns,selected.data()+row*output.columns);
        hostInputs[entry.first]=std::move(selected);
    }
    for(auto& op:ops_)if(op.kind==Kind::KVUpdate||op.kind==Kind::KVChunk) {
        const auto& positions=hostInputs.at(op.arguments[4]);const auto& slots=hostInputs.at(op.arguments[2]);
        const auto& active=hostInputs.at(op.arguments[3]);
        for(size_t row=0;row<active.size();++row) {
            if(!std::isfinite(active[row])||(active[row]!=0&&active[row]!=1)||!std::isfinite(slots[row])||slots[row]<0||
                std::floor(slots[row])!=slots[row]||(active[row]!=0&&slots[row]>=op.dimensions[2]))
                throw std::runtime_error("Metal attention active row/slot invalid");
            for(size_t axis=0;axis<3;++axis)if(!std::isfinite(positions[row*3+axis]))
                throw std::runtime_error("Metal RoPE position must be finite");
        }
        if(op.kind==Kind::KVChunk) {
            // Direct slot lookup requires a contiguous active prefix in every
            // row. Overwrites, gaps and paused/resumed rows keep the scan.
            const size_t chunk=op.dimensions[3];op.contiguousSlots=true;
            for(size_t base=0;base<active.size();base+=chunk) {
                bool paused=false;
                for(size_t t=0;t<chunk;++t) {
                    if(active[base+t]==0)paused=true;
                    else if(paused||slots[base+t]!=slots[base]+float(t))op.contiguousSlots=false;
                }
            }
        }
    }
    for(size_t i=0;i<inputs_.size();++i){
        if(skip.count(i))continue;
        X::Tensor tensor(inputs.Get(i));auto lease=tensor.Acquire();auto info=tensor.Info();auto& slot=slots_[inputs_[i]];
        MaterializeDefault(inputs_[i]);
        gpu_->Update(slot.matrix,static_cast<const float*>(info.data),slot.rows*slot.columns);
    }
}
bool MetalCompiledGraph::SwapOutputStorage(size_t part,MetalMatrix& replacement) {
    if(part>=outputStorageGroups_.size()||outputStorageGroups_[part].empty())return false;
    const auto finished=slots_[outputParts_[part]].matrix;
    if(!finished.CompatibleLayout(replacement))return false;
    for(auto id:outputStorageGroups_[part])slots_[id].matrix=replacement;
    replacement=finished;return true;
}
void MetalCompiledGraph::EncodeGreedy(MetalContext& gpu,const MetalMatrix& prediction) {
#if defined(GARNET_METAL_FUSED_HEAD_ARGMAX)||defined(GARNET_METAL_SCREENED_HEAD)
    Encode(gpu,true,&prediction);
#else
    Encode(gpu);gpu.EncodeArgmax(slots_[output_].matrix,prediction);
#endif
}
void MetalCompiledGraph::Encode(MetalContext& gpu,bool packedOutput,const MetalMatrix* greedyOutput) {
    bool greedyEncoded=false;
    // Opt-in isolated-operation diagnostics deliberately split submissions.
    // They identify expensive kernels; their latency is not a serving metric.
    const char* profile=std::getenv("GARNET_METAL_PROFILE_OPS");
    const bool profileOps=profile&&std::strcmp(profile,"1")==0;
    static const char* names[]={"MatMul","BF16MatVec","Norm","Silu","Add","Bias","Mask","Conv","Delta","Slice","Prepare","GateNorm","Concat","NormRope","KVUpdate","KVChunk","AttentionChunk","Scores","AttentionContext","Vision","Layout","DeltaSplit","ConvChunk","DeltaChunk"};
    for(size_t i=0;i<ops_.size();++i) {
        if(!packedOutput&&!residentOps_[i])continue;
        const auto& op=ops_[i];
        auto a=slots_[op.a].matrix,b=slots_[op.b].matrix,out=slots_[op.out].matrix;
#ifdef GARNET_METAL_SCREENED_HEAD
        if(greedyOutput&&!profileOps&&i+1==ops_.size()&&op.out==output_&&op.kind==Kind::BF16MatVec&&
            slots_[op.b].bf16.Valid()&&gpu.EncodeScreenedArgmax(a,slots_[op.b].bf16,*greedyOutput)){
            greedyEncoded=true;continue;
        }
#endif
#ifdef GARNET_METAL_FUSED_HEAD_ARGMAX
        if(greedyOutput&&!profileOps&&i+1==ops_.size()&&op.out==output_&&
            (op.kind==Kind::BF16MatVec||(op.kind==Kind::MatMul&&op.transpose))&&
            slots_[op.b].bf16.Valid()&&gpu.EncodeBF16Argmax(a,slots_[op.b].bf16,*greedyOutput)) {
            greedyEncoded=true;continue;
        }
#endif
#ifdef GARNET_METAL_PACKED_PREFILL_NORM
        if(!profileOps&&op.kind==Kind::Add&&i+2<ops_.size()&&
            (packedOutput||(residentOps_[i+1]&&residentOps_[i+2]))) {
            const auto& layout=ops_[i+1];const auto& norm=ops_[i+2];
            const auto& packed=slots_[op.out];const auto& reshaped=slots_[layout.out];
            if(layout.kind==Kind::Layout&&layout.a==op.out&&layout.dimensions.size()==1&&!layout.dimensions[0]&&
                norm.kind==Kind::Norm&&norm.a==layout.out&&packed.rows<=2&&reshaped.rows>packed.rows&&
                reshaped.columns&&packed.columns%reshaped.columns==0&&
                packed.columns/reshaped.columns<=kMetalPrefillChunkLimit&&a.CompatibleLayout(b)&&a.CompatibleLayout(out)&&
                reshaped.matrix.CompatibleLayout(slots_[norm.out].matrix)) {
                gpu.EncodePackedPrefillNorm(a,b,slots_[norm.b].matrix,out,reshaped.matrix,slots_[norm.out].matrix,norm.epsilon,norm.offset);
                i+=2;continue;
            }
        }
#endif
#ifdef GARNET_METAL_RESIDUAL_NORM
        if(!profileOps&&op.kind==Kind::Add&&i+1<ops_.size()&&(packedOutput||residentOps_[i+1])) {
            const auto& next=ops_[i+1];
            if(next.kind==Kind::Norm&&next.a==op.out&&a.CompatibleLayout(b)&&a.CompatibleLayout(out)&&
                a.CompatibleLayout(slots_[next.out].matrix)) {
                gpu.EncodeResidualRMSNorm(a,b,slots_[next.b].matrix,out,slots_[next.out].matrix,next.epsilon,next.offset);
                ++i;continue;
            }
        }
#endif
#ifdef GARNET_METAL_MLP_PREFILL_GRAPH
        if(!profileOps&&slots_[op.a].rows>2&&i+3<ops_.size()&&
            (packedOutput||(residentOps_[i+1]&&residentOps_[i+2]&&residentOps_[i+3]))) {
            const auto& up=ops_[i+1];const auto& silu=ops_[i+2];const auto& down=ops_[i+3];
            auto projection=[&](const Op& value){return (value.kind==Kind::BF16MatVec||
                (value.kind==Kind::MatMul&&value.transpose))&&slots_[value.b].bf16.Valid();};
            if(projection(op)&&projection(up)&&projection(down)&&op.a==up.a&&op.out!=up.out&&
                silu.kind==Kind::Silu&&silu.a==op.out&&silu.b==up.out&&down.a==silu.out&&
                gpu.EncodeMlpPrefillGraph(a,slots_[op.b].bf16,slots_[up.b].bf16,
                    slots_[down.b].bf16,slots_[down.out].matrix)) {
                i+=3;continue;
            }
        }
#endif
#ifdef GARNET_METAL_JOINED_GLOBAL_MLP
        if(!profileOps&&slots_[op.a].rows>2&&i+2<ops_.size()&&
            (packedOutput||(residentOps_[i+1]&&residentOps_[i+2]))) {
            const auto& next=ops_[i+1];const auto& silu=ops_[i+2];
            auto projection=[&](const Op& v){return (v.kind==Kind::BF16MatVec||
                (v.kind==Kind::MatMul&&v.transpose))&&slots_[v.b].bf16.Valid();};
            if(projection(op)&&projection(next)&&op.a==next.a&&op.out!=next.out&&
                silu.kind==Kind::Silu&&silu.a==op.out&&silu.b==next.out&&
                gpu.EncodeJoinedGlobalMlp(a,slots_[op.b].bf16,slots_[next.b].bf16,
                    out,slots_[next.out].matrix,slots_[silu.out].matrix)) {
                i+=2;continue;
            }
        }
#endif
#ifdef GARNET_METAL_JOINED_PREFILL
        if(!profileOps&&slots_[op.a].rows>2&&i+1<ops_.size()&&(packedOutput||residentOps_[i+1])) {
            const auto& next=ops_[i+1];
            auto projection=[&](const Op& value){return (value.kind==Kind::BF16MatVec||
                (value.kind==Kind::MatMul&&value.transpose))&&slots_[value.b].bf16.Valid();};
            if(projection(op)&&projection(next)&&op.a==next.a&&op.out!=next.out&&
                gpu.EncodeJoinedBF16Prefill(a,slots_[op.b].bf16,slots_[next.b].bf16,out,slots_[next.out].matrix)) {
                ++i;continue;
            }
        }
#endif
        if(op.kind==Kind::MatMul) {
            if(slots_[op.b].bf16.Valid()){if(!op.transpose)throw std::logic_error("BF16 projection must transpose checkpoint rows");gpu.EncodeBF16MatVec(a,slots_[op.b].bf16,out);}
            else gpu.EncodeMatMul(a,b,out,op.transpose);
        }
        else if(op.kind==Kind::BF16MatVec)gpu.EncodeBF16MatVec(a,slots_[op.b].bf16,out);
        else if(op.kind==Kind::Layout)gpu.EncodeLayout(a,b,out,op.dimensions[0]);
        else if(op.kind==Kind::Norm)gpu.EncodeRMSNorm(a,b,out,op.epsilon,op.offset);
        else if(op.kind==Kind::Silu)gpu.EncodeSiluMul(a,b,out);
        else if(op.kind==Kind::Add)gpu.EncodeAdd(a,b,out);
        else if(op.kind==Kind::Bias)gpu.EncodeBias(a,b,out);
        else if(op.kind==Kind::Mask)gpu.EncodeMaskRows(a,b,out);
        else if(op.kind==Kind::Conv) {
            auto& args=op.arguments;gpu.EncodeCausalConv(slots_[args[0]].matrix,slots_[args[1]].matrix,
                slots_[args[2]].matrix,slots_[args[3]].matrix,out,op.dimensions[0]);
        }else if(op.kind==Kind::ConvChunk) {
            auto& args=op.arguments;gpu.EncodeCausalConvChunk(slots_[args[0]].matrix,slots_[args[1]].matrix,
                slots_[args[2]].matrix,slots_[args[3]].matrix,out,slots_[op.auxiliaryOutput].matrix,op.dimensions[0],op.dimensions[1]);
        }else if(op.kind==Kind::DeltaChunk) {
            auto& args=op.arguments;gpu.EncodeGatedDeltaChunk(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                out,slots_[op.auxiliaryOutput].matrix,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]);
        }else if(op.kind==Kind::Delta) {
            auto& args=op.arguments;gpu.EncodeGatedDelta(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,slots_[args[4]].matrix,slots_[args[5]].matrix,slots_[args[6]].matrix,out,
                op.dimensions[0],op.dimensions[1],op.dimensions[2]);
        }else if(op.kind==Kind::DeltaSplit) {
            auto& args=op.arguments;gpu.EncodeGatedDeltaSplit(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,slots_[args[4]].matrix,slots_[args[5]].matrix,slots_[args[6]].matrix,out,
                slots_[op.auxiliaryOutput].matrix,op.dimensions[0],op.dimensions[1],op.dimensions[2]);
        }else if(op.kind==Kind::Prepare) {
            auto& args=op.arguments;gpu.EncodeLinearPrepare(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,slots_[args[4]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4]);
        }else if(op.kind==Kind::GateNorm) {
            auto& args=op.arguments;gpu.EncodeGatedNorm(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                out,op.dimensions[0],op.dimensions[1],op.epsilon,op.dimensions.size()>2?op.dimensions[2]:0);
        }else if(op.kind==Kind::Concat) {
            auto& args=op.arguments;gpu.EncodeConcat(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,out,op.transpose);
        }else if(op.kind==Kind::NormRope) {
            auto& args=op.arguments;gpu.EncodeNormRope(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.epsilon,op.offset,op.dimensions[3]!=0);
        }else if(op.kind==Kind::KVChunk) {
            auto& args=op.arguments;gpu.EncodeKVChunk(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.contiguousSlots);
        }else if(op.kind==Kind::AttentionChunk) {
            std::vector<MetalMatrix> args;for(auto id:op.arguments)args.push_back(slots_[id].matrix);
            gpu.EncodeAttentionChunk(args,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4],op.dimensions[5]!=0);
        }else if(op.kind==Kind::KVUpdate) {
            auto& args=op.arguments;gpu.EncodeKVUpdate(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2]);
        }else if(op.kind==Kind::Scores) {
            auto& args=op.arguments;gpu.EncodeAttentionScores(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]);
        }else if(op.kind==Kind::Vision) {
            std::vector<MetalMatrix> args;for(auto id:op.arguments)args.push_back(slots_[id].matrix);
            gpu.EncodeVision(op.dimensions[0],args,out,op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4]);
        }else if(op.kind==Kind::AttentionContext) {
            auto& args=op.arguments;gpu.EncodeAttentionContext(slots_[args[0]].matrix,slots_[args[1]].matrix,slots_[args[2]].matrix,
                slots_[args[3]].matrix,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3],op.dimensions[4]!=0);
        }else {
            gpu.EncodeSliceGroups(a,out,op.dimensions[0],op.dimensions[1],op.dimensions[2],op.dimensions[3]);
        }
        if(profileOps) {
            gpu.SubmitAndWait();
            std::fprintf(stderr,"metal_op_profile {\"kind\":\"%s\",\"a_rows\":%zu,\"a_cols\":%zu,\"out_rows\":%zu,\"out_cols\":%zu,\"bf16\":%s,\"exact_half_prefill\":%s,\"gpu_ms\":%.9g,\"serialized_diagnostic\":true}\n",
                names[size_t(op.kind)],slots_[op.a].rows,slots_[op.a].columns,
                slots_[op.out].rows,slots_[op.out].columns,
                slots_[op.b].bf16.Valid()?"true":"false",
                slots_[op.a].rows>2&&slots_[op.b].bf16.ExactHalfReady()?"true":"false",gpu.LastGPUMs());
            gpu.Begin();
        }
    }
    if(greedyOutput&&!greedyEncoded)gpu.EncodeArgmax(slots_[output_].matrix,*greedyOutput);
}
X::Value MetalCompiledGraph::Run(X::Value inputs) {
    std::lock_guard<std::mutex> lock(mutex_);
    visionParallelDispatches_=0;
    visionPrefetchDispatches_=0;visionVectorDispatches_=0;mlpGraphDispatches_=0;
    visionGlobalAlphaDispatches_=0;
    auto start=std::chrono::steady_clock::now();
    PrepareInputs(inputs);
    MaterializeDefault(output_);
    gpu_->Begin();
    try {
        Encode(*gpu_);
        gpu_->SubmitAndWait();
        visionGlobalAlphaDispatches_=gpu_->LastVisionGlobalAlphaDispatches();
        visionParallelDispatches_=gpu_->LastVisionParallelDispatches();
        visionPrefetchDispatches_=gpu_->LastVisionPrefetchDispatches();
        visionVectorDispatches_=gpu_->LastVisionVectorDispatches();
        mlpGraphDispatches_=gpu_->LastMlpGraphDispatches();
    }catch(...){gpu_->Abort();throw;}
    auto result=gpu_->Download(slots_[output_].matrix);auto& output=slots_[output_];
    inferenceMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return X::Tensor::Create(Host(),X3_TENSOR_FLOAT32,output.shape,result.data(),result.size()*4);
}
}

namespace Garnet {
std::vector<MetalResourceReference> MetalCompiledGraph::MemoryReferences() {
    std::lock_guard<std::mutex> lock(mutex_);if(!ready_)throw std::logic_error("Inventory requires a ready graph");
    std::vector<MetalResourceReference> result;std::vector<bool> active(slots_.size(),false),input(slots_.size(),false),output(slots_.size(),false);
    auto mark=[&](size_t i){if(i<active.size())active[i]=true;};
    std::vector<bool> anyWriter(slots_.size(),false),residentWriter(slots_.size(),false);
    for(size_t i=0;i<ops_.size();++i){bool resident=residentOps_.size()!=ops_.size()||residentOps_[i];auto write=[&](size_t slot){if(slot<slots_.size()){anyWriter[slot]=true;if(resident)residentWriter[slot]=true;}};write(ops_[i].out);write(ops_[i].auxiliaryOutput);}

    for(size_t i=0;i<ops_.size();++i)if(residentOps_.size()!=ops_.size()||residentOps_[i]) {
        const auto& op=ops_[i];mark(op.a);mark(op.b);mark(op.out);mark(op.auxiliaryOutput);for(auto j:op.arguments)mark(j);
    }
    for(auto i:inputs_)if(i<input.size())input[i]=true;for(auto i:outputParts_)if(i<output.size())output[i]=true;
    std::vector<std::string> names(slots_.size());std::vector<std::pair<std::string,size_t>> sorted(weights_.begin(),weights_.end());std::sort(sorted.begin(),sorted.end());
    for(const auto& entry:sorted)if(entry.second<names.size()){if(!names[entry.second].empty())names[entry.second]+=";";names[entry.second]+=entry.first;}
    auto append=[&](const Slot& slot,size_t i,const std::string& role,const std::string& name,bool weight,bool in,bool out,bool resident,bool legacy){
        auto emit=[&](const MetalBufferInfo& info){if(!info.identity)return;MetalResourceReference r;r.buffer=info;r.slot=i;r.role=role;r.name=name;r.shape=slot.shape;r.weight=weight;r.input=in;r.outputPart=out;r.residentOperand=resident;r.anyWriter=i<anyWriter.size()&&anyWriter[i];r.residentWriter=i<residentWriter.size()&&residentWriter[i];r.legacyOutput=legacy;result.push_back(std::move(r));};
        emit(MetalContext::MatrixBufferInfo(slot.matrix));for(const auto& info:MetalContext::WeightBufferInfo(slot.bf16))emit(info);
    };
    for(size_t i=0;i<slots_.size();++i)append(slots_[i],i,"slot",names[i],!names[i].empty(),input[i],output[i],active[i],i==output_);
    auto held=[&](const auto& records,const char* role){std::vector<std::string> keys;for(const auto& entry:records)keys.push_back(entry.first);std::sort(keys.begin(),keys.end());for(const auto& key:keys)append(records.at(key).slot,SIZE_MAX,role,key,true,false,false,false,false);};
    held(sharedWeights_,"shared_weight_record");held(weightRecords_,"weight_record");return result;
}
}

namespace Garnet {
void MetalCompiledGraph::RetireResidentDefaults() {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    if(!lazyDefaultPlanReady_) {
        if(!ready_||outputParts_.size()!=3||inputs_.size()<3||residentOps_.size()!=ops_.size())return;
        std::vector<MetalBufferInfo> infos;std::unordered_map<uintptr_t,size_t> counts;std::unordered_set<uintptr_t> protectedWeights;
        for(const auto& slot:slots_){auto info=MetalContext::MatrixBufferInfo(slot.matrix);infos.push_back(info);if(info.identity)++counts[info.identity];for(const auto& b:MetalContext::WeightBufferInfo(slot.bf16))protectedWeights.insert(b.identity);}
        for(const auto& record:weightRecords_){auto info=MetalContext::MatrixBufferInfo(record.second.slot.matrix);if(info.identity)protectedWeights.insert(info.identity);for(const auto& b:MetalContext::WeightBufferInfo(record.second.slot.bf16))protectedWeights.insert(b.identity);}
        auto eligible=[&](size_t id){const auto& slot=slots_[id];const auto& info=infos[id];return info.identity&&counts[info.identity]==1&&!protectedWeights.count(info.identity)&&gpu_->IsDefaultLayout(slot.matrix,slot.rows,slot.columns);};
        std::vector<size_t> plan;
        for(size_t part=1;part<=2;++part){auto id=inputs_[part];if(inputShapes_[part].size()<3||std::find(outputParts_.begin(),outputParts_.end(),id)!=outputParts_.end())continue;if(eligible(id))plan.push_back(id);}
        if(!ops_.empty()&&ops_.back().kind==Kind::Concat&&ops_.back().out==output_&&!residentOps_.back()&&std::find(inputs_.begin(),inputs_.end(),output_)==inputs_.end()&&std::find(outputParts_.begin(),outputParts_.end(),output_)==outputParts_.end()&&eligible(output_))plan.push_back(output_);
        lazyDefaultSlots_=std::move(plan);lazyDefaultPlanReady_=true;
    }
    for(auto id:lazyDefaultSlots_)if(slots_[id].matrix.Valid()) {auto info=MetalContext::MatrixBufferInfo(slots_[id].matrix);lazyRetiredBytes_+=info.allocatedBytes;slots_[id].matrix=MetalMatrix();}
#endif
}
void MetalCompiledGraph::MaterializeDefault(size_t id) {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    if(slots_[id].matrix.Valid())return;
    if(std::find(lazyDefaultSlots_.begin(),lazyDefaultSlots_.end(),id)==lazyDefaultSlots_.end())throw std::logic_error("Missing non-deferred graph storage");
    auto matrix=gpu_->Allocate(slots_[id].rows,slots_[id].columns);auto info=MetalContext::MatrixBufferInfo(matrix);slots_[id].matrix=std::move(matrix);lazyRehydratedBytes_+=info.allocatedBytes;
#else
    (void)id;
#endif
}
size_t MetalCompiledGraph::LazyRetiredBytes() const {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    std::lock_guard<std::mutex> lock(mutex_);return lazyRetiredBytes_;
#else
    return 0;
#endif
}
size_t MetalCompiledGraph::LazyRehydratedBytes() const {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    std::lock_guard<std::mutex> lock(mutex_);return lazyRehydratedBytes_;
#else
    return 0;
#endif
}
size_t MetalCompiledGraph::LazyPlannedSlots() const {
#ifdef GARNET_METAL_LAZY_RESIDENT_DEFAULTS
    std::lock_guard<std::mutex> lock(mutex_);return lazyDefaultSlots_.size();
#else
    return 0;
#endif
}
}
