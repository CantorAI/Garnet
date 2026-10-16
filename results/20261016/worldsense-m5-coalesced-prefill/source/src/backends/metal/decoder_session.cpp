// SPDX-License-Identifier: Apache-2.0
#include "decoder_session.h"
#include "lowering_context.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
namespace Garnet {
namespace {
MetalCompiledGraph* Graph(const X::Value& value,X3PackageHost* host) {
    if(value.runtime()!=host->runtime)throw std::invalid_argument("Metal session graph runtime mismatch");
    auto graph=value.NativeData<MetalCompiledGraph>();
    if(!graph)throw std::invalid_argument("Metal session requires compiled_graph instances");
    return graph;
}
X::Value TensorValue(X3PackageHost* host,MetalContext& gpu,const MetalMatrix& matrix,const std::vector<int64_t>& shape) {
    auto data=gpu.Download(matrix);
    return X::Tensor::Create(host,X3_TENSOR_FLOAT32,shape,data.data(),data.size()*sizeof(float));
}
}
void MetalDecoderSession::Clear() {
    screenedHeadDispatches_=0;screenedHeadSeedRows_=0;interleavedScreenDispatches_=0;
    reusedSourceBytes_=reusedCopyBytes_=reusedCopyStates_=0;
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
    std::vector<float>().swap(resetZeroSource_);
#endif
    prefillPartialSubmissions_=0;fusedHeadArgmaxDispatches_=0;bodyShapeDispatches_=bodyShapeMisses_=0;deferredZeroBytes_=deferredZeroStates_=0;
    dirtyPrefixResetBytes_=dirtyPrefixResetStates_=dirtyPrefixResetFallbackStates_=0;
    joinedPrefillPairs_=0;
    constantHeadDispatches_=0;singleCoderHeadDispatches_=0;residualNormDispatches_=0;packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;wideBatchPrefillDispatches_=0;interleavedBodyDispatches_=0;coalescedPrefillDispatches_=0;coalescedBlockDispatches_=0;mlpGraphDispatches_=0;joinedGlobalMlpBlocks_=0;
    hidden_=MetalMatrix();prediction_=MetalMatrix();hiddenShape_.clear();states_.clear();owners_.clear();head_=nullptr;
    gpu_.reset();kvInplaceUpdates_=0;poisoned_=false;inferenceMs_=0;prepareMs_=encodeMs_=gpuMs_=waitMs_=0;
}
void MetalDecoderSession::Close() {std::lock_guard<std::mutex> lock(mutex_);Clear();}
void MetalDecoderSession::ResetUnlocked() {
    screenedHeadDispatches_=0;screenedHeadSeedRows_=0;interleavedScreenDispatches_=0;
    reusedCopyBytes_=reusedCopyStates_=0;
    prefillPartialSubmissions_=0;fusedHeadArgmaxDispatches_=0;bodyShapeDispatches_=bodyShapeMisses_=0;deferredZeroBytes_=deferredZeroStates_=0;
    joinedPrefillPairs_=0;
    constantHeadDispatches_=0;singleCoderHeadDispatches_=0;residualNormDispatches_=0;packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;wideBatchPrefillDispatches_=0;interleavedBodyDispatches_=0;coalescedPrefillDispatches_=0;coalescedBlockDispatches_=0;mlpGraphDispatches_=0;joinedGlobalMlpBlocks_=0;
    if(!gpu_)throw std::logic_error("Metal session is closed");
    dirtyPrefixResetBytes_=dirtyPrefixResetStates_=dirtyPrefixResetFallbackStates_=0;
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
    const bool forceFull=poisoned_;poisoned_=true;
#endif
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
    poisoned_=true;
#endif
    for(auto& state:states_)for(size_t i=0;i<2;++i) {
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
        const size_t count=state.rows*state.columns[i];
        if(count>resetZeroSource_.size())throw std::logic_error("Retained reset source does not cover private state");
        gpu_->Update(state.matrix[i],resetZeroSource_.data(),count);
        reusedCopyBytes_+=count*sizeof(float);++reusedCopyStates_;
#else
#ifdef GARNET_METAL_DEFERRED_STATE_ZERO
        gpu_->DeferZero(state.matrix[i]);
#else
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
        if(!forceFull&&state.prefixLayout[i]&&state.zeroOutsideDirty[i]) {
            const auto& shape=state.shape[i];
            gpu_->ZeroPrefixGroups(state.matrix[i],size_t(shape[1]),size_t(shape[2]),size_t(shape[3]),state.dirtyExtent[i]);
            dirtyPrefixResetBytes_+=state.rows*size_t(shape[1])*state.dirtyExtent[i]*size_t(shape[3])*sizeof(float);
            ++dirtyPrefixResetStates_;
        } else {
            gpu_->Update(state.matrix[i],std::vector<float>(state.rows*state.columns[i],0));
            dirtyPrefixResetBytes_+=state.rows*state.columns[i]*sizeof(float);++dirtyPrefixResetFallbackStates_;
        }
        state.dirtyExtent[i]=0;state.zeroOutsideDirty[i]=state.prefixLayout[i];
#else
#ifdef GARNET_METAL_DIRECT_STATE_ZERO
        gpu_->Zero(state.matrix[i]);
#else
        gpu_->Update(state.matrix[i],std::vector<float>(state.rows*state.columns[i],0));
#endif
#endif
#endif
#endif
    }
    hidden_=MetalMatrix();hiddenShape_.clear();kvInplaceUpdates_=0;poisoned_=false;inferenceMs_=0;prepareMs_=encodeMs_=gpuMs_=waitMs_=0;
}
bool MetalDecoderSession::SetCoalescedPrefill(bool enabled) {std::lock_guard<std::mutex> lock(mutex_);if(!gpu_)throw std::logic_error("Metal session is closed");return gpu_->SetCoalescedPrefill(enabled);}
void MetalDecoderSession::Reset() {std::lock_guard<std::mutex> lock(mutex_);ResetUnlocked();}
bool MetalDecoderSession::Build(X::Value graphs,X::Value head) {
    std::lock_guard<std::mutex> lock(mutex_);Clear();
    if(graphs.runtime()!=Host()->runtime||!graphs.IsList()||graphs.Size()==0)
        throw std::invalid_argument("Metal session requires a nonempty graph list");
    try {
        gpu_=std::make_unique<MetalContext>();
        size_t width=0,batch=0;
        for(uint64_t i=0;i<graphs.Size();++i) {
            auto owner=graphs.Get(i);auto graph=Graph(owner,Host());
            std::lock_guard<std::mutex> graphLock(graph->mutex_);
            if(!graph->ready_||graph->inputs_.size()<3||graph->outputParts_.size()!=3)
                throw std::invalid_argument("Metal session requires packed hidden and two-state layers");
            auto& input=graph->slots_[graph->inputs_[0]];
            if(input.shape.size()!=2&&input.shape.size()!=3)throw std::invalid_argument("Metal session hidden rank invalid");
            const auto layerWidth=size_t(input.shape.back());
            if(i&&(input.rows!=batch||layerWidth!=width))throw std::invalid_argument("Metal session layer dimensions mismatch");
            batch=input.rows;width=layerWidth;State state;state.rows=batch;
            for(size_t k=0;k<2;++k) {
                auto& source=graph->slots_[graph->inputs_[k+1]];
                auto& output=graph->slots_[graph->outputParts_[k+1]];
                if(source.rows!=batch||output.rows!=source.rows||output.columns!=source.columns)
                    throw std::invalid_argument("Metal session state layout mismatch");
                state.shape[k]=source.shape;state.columns[k]=source.columns;
                state.matrix[k]=gpu_->Allocate(source.rows,source.columns);
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
                state.prefixLayout[k]=graph->CanResetKVPrefix(k+1);
#endif
            }
            states_.push_back(std::move(state));owners_.push_back(std::move(owner));
        }
        if(head.IsValid()&&!head.IsNone()) {
            head_=Graph(head,Host());std::lock_guard<std::mutex> graphLock(head_->mutex_);
            if(!head_->ready_||head_->inputs_.size()!=1||head_->inputShapes_[0]!=std::vector<int64_t>{int64_t(batch),int64_t(width)})
                throw std::invalid_argument("Metal session vocabulary head dimensions mismatch");
            headGeneration_=head_->generation_;owners_.push_back(head);prediction_=gpu_->Allocate(batch,2);
        }
#ifdef GARNET_METAL_REUSED_ZERO_SOURCE
        size_t maximum=0;
        for(const auto& state:states_)for(size_t k=0;k<2;++k)maximum=std::max(maximum,state.rows*state.columns[k]);
        resetZeroSource_.assign(maximum,0.0f);reusedSourceBytes_=maximum*sizeof(float);
#endif
        ResetUnlocked();return true;
    }catch(...){Clear();throw;}
}
X::Value MetalDecoderSession::Run(X::Value graphs,X::Value hidden,X::Value positions,
    X::Value slots,X::Value active,int64_t predictionToken) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_)throw std::logic_error("Reset a closed/failed Metal session before running");
    if(graphs.runtime()!=Host()->runtime||!graphs.IsList()||graphs.Size()!=states_.size())
        throw std::invalid_argument("Metal session layer count mismatch");
    std::vector<MetalCompiledGraph*> layers,locked;
    for(uint64_t i=0;i<graphs.Size();++i)layers.push_back(Graph(graphs.Get(i),Host()));
    locked=layers;if(head_)locked.push_back(head_);
    std::sort(locked.begin(),locked.end(),std::less<MetalCompiledGraph*>());
    if(std::adjacent_find(locked.begin(),locked.end())!=locked.end())throw std::invalid_argument("Metal session graphs must be distinct");
    std::vector<std::unique_lock<std::mutex>> graphLocks;
    for(auto graph:locked)graphLocks.emplace_back(graph->mutex_);
    if(head_&&(!head_->ready_||head_->generation_!=headGeneration_))throw std::logic_error("Metal session head was closed/rebuilt");
    const auto start=std::chrono::steady_clock::now();
    const auto& inputShape=layers[0]->inputShapes_;
    if(inputShape.empty()||(inputShape[0].size()!=2&&inputShape[0].size()!=3))throw std::invalid_argument("Metal session hidden profile missing");
    const size_t chunk=inputShape[0].size()==3?size_t(inputShape[0][1]):1;
    if(predictionToken<-1||(predictionToken>=0&&(!head_||uint64_t(predictionToken)>=chunk)))
        throw std::invalid_argument("Metal session prediction token/head invalid");
    // Validate and upload every host input before encoding or changing state.
    for(size_t i=0;i<layers.size();++i) {
        auto graph=layers[i];const auto& state=states_[i];
        if(!graph->ready_||graph->outputParts_.size()!=3||graph->inputShapes_.size()<3||graph->inputShapes_[0]!=inputShape[0]||
            graph->inputShapes_[1]!=state.shape[0]||graph->inputShapes_[2]!=state.shape[1])
            throw std::invalid_argument("Metal session layer/state profile mismatch");
        auto inputs=X::Value::List(Host());inputs.Append(i?X::Value():hidden);inputs.Append(X::Value());inputs.Append(X::Value());
        if(graph->inputs_.size()==4)inputs.Append(active);
        else if(graph->inputs_.size()==6){inputs.Append(positions);inputs.Append(slots);inputs.Append(active);}
        else throw std::invalid_argument("Metal session unsupported controls");
        graph->PrepareInputs(inputs,i?std::unordered_set<size_t>{0,1,2}:std::unordered_set<size_t>{1,2});
    }
    {
        X::Tensor flags(active);auto info=flags.Info();auto lease=flags.Acquire();
        const auto* data=static_cast<const float*>(info.data);
        const size_t count=states_[0].rows*chunk;
        for(size_t i=0;i<count;++i)if(!std::isfinite(data[i])||(data[i]!=0&&data[i]!=1))
            throw std::invalid_argument("Invalid Metal session active flags");
    }
#ifdef GARNET_METAL_DIRTY_PREFIX_RESET
    // Host controls were validated above. Mark before submission so a failed
    // command always takes the existing full-reset recovery path.
    size_t writeExtent=0;bool needExtent=false;
    for(size_t i=0;i<states_.size();++i)for(size_t k=0;k<2;++k)
        needExtent=needExtent||(states_[i].prefixLayout[k]&&states_[i].zeroOutsideDirty[k]&&layers[i]->CanResetKVPrefix(k+1));
    if(needExtent) {
        X::Tensor slotTensor(slots),flagTensor(active);auto sl=slotTensor.Acquire();auto fl=flagTensor.Acquire();
        const auto* slotData=static_cast<const float*>(slotTensor.Info().data);
        const auto* flagData=static_cast<const float*>(flagTensor.Info().data);
        for(size_t j=0;j<states_[0].rows*chunk;++j)if(flagData[j]!=0)writeExtent=std::max(writeExtent,size_t(slotData[j])+1);
    }
    for(size_t i=0;i<layers.size();++i)for(size_t k=0;k<2;++k) {
        auto& state=states_[i];
        if(!state.prefixLayout[k]||!state.zeroOutsideDirty[k])continue;
        if(!layers[i]->CanResetKVPrefix(k+1)){state.zeroOutsideDirty[k]=false;continue;}
        state.dirtyExtent[k]=std::max(state.dirtyExtent[k],writeExtent);
    }
#endif
    if(hiddenShape_!=inputShape[0]) {
        const auto& slot=layers[0]->slots_[layers[0]->inputs_[0]];
        hidden_=gpu_->Allocate(slot.rows,slot.columns);
    }
    prefillPartialSubmissions_=0;fusedHeadArgmaxDispatches_=0;bodyShapeDispatches_=bodyShapeMisses_=0;deferredZeroBytes_=deferredZeroStates_=0;
    screenedHeadDispatches_=0;screenedHeadSeedRows_=0;interleavedScreenDispatches_=0;
    const auto prepared=std::chrono::steady_clock::now();
    prepareMs_=std::chrono::duration<double,std::milli>(prepared-start).count();
    kvInplaceUpdates_=0;constantHeadDispatches_=0;singleCoderHeadDispatches_=0;residualNormDispatches_=0;packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;wideBatchPrefillDispatches_=0;interleavedBodyDispatches_=0;coalescedPrefillDispatches_=0;coalescedBlockDispatches_=0;mlpGraphDispatches_=0;joinedGlobalMlpBlocks_=0;poisoned_=true;gpu_->Begin();
    try {
        auto current=layers[0]->slots_[layers[0]->inputs_[0]].matrix;
        for(size_t i=0;i<layers.size();++i) {
            auto graph=layers[i];std::array<MetalMatrix,3> saved;
            std::array<bool,2> inplace{false,false};
            std::vector<std::pair<size_t,MetalMatrix>> savedOutputs;
            for(size_t k=0;k<2;++k)if(graph->CanUpdateKVInPlace(k+1)) {
                const auto& group=graph->outputStorageGroups_[k+1];
                if(!states_[i].matrix[k].CompatibleLayout(graph->slots_[graph->outputParts_[k+1]].matrix))
                    throw std::logic_error("Private KV cache layout mismatch");
                for(auto id:group){savedOutputs.emplace_back(id,graph->slots_[id].matrix);}
                inplace[k]=true;
            }
            for(size_t k=0;k<2;++k)if(inplace[k])for(auto id:graph->outputStorageGroups_[k+1])
                graph->slots_[id].matrix=states_[i].matrix[k];
            for(size_t k=0;k<3;++k)saved[k]=graph->slots_[graph->inputs_[k]].matrix;
            graph->slots_[graph->inputs_[0]].matrix=current;
            graph->slots_[graph->inputs_[1]].matrix=states_[i].matrix[0];
            graph->slots_[graph->inputs_[2]].matrix=states_[i].matrix[1];
            auto restore=[&](){
                for(size_t k=0;k<3;++k)graph->slots_[graph->inputs_[k]].matrix=saved[k];
                for(auto& entry:savedOutputs)graph->slots_[entry.first].matrix=entry.second;
            };
            try {graph->Encode(*gpu_,false);}
            catch(...){restore();throw;}
            restore();
            kvInplaceUpdates_+=size_t(inplace[0])+size_t(inplace[1]);
            current=graph->slots_[graph->outputParts_[0]].matrix;
            // Keep the new state private to this session without copying it
            // back. The previous state becomes this graph's next output
            // storage; graph locks cover the exchange and GPU completion.
            for(size_t k=0;k<2;++k)if(!inplace[k]&&!graph->SwapOutputStorage(k+1,states_[i].matrix[k]))
                gpu_->EncodeSliceGroups(graph->slots_[graph->outputParts_[k+1]].matrix,
                    states_[i].matrix[k],1,states_[i].columns[k],0,states_[i].columns[k]);
#ifdef GARNET_METAL_PREFILL_STREAM
            if(chunk>1&&(i+1)%4==0&&i+1<layers.size())gpu_->Flush();
#endif
        }
        const size_t hiddenWidth=size_t(inputShape[0].back())*chunk;
        gpu_->EncodeSliceGroups(current,hidden_,1,hiddenWidth,0,hiddenWidth);
        if(predictionToken>=0) {
            auto& headInput=head_->slots_[head_->inputs_[0]];
            gpu_->EncodeSliceGroups(current,headInput.matrix,1,chunk*headInput.columns,
                size_t(predictionToken)*headInput.columns,headInput.columns);
#ifdef GARNET_METAL_FUSED_HEAD_DECODE_ONLY
            if(chunk>1) {
                head_->Encode(*gpu_);
                gpu_->EncodeArgmax(head_->slots_[head_->output_].matrix,prediction_);
            } else
#endif
            head_->EncodeGreedy(*gpu_,prediction_);
        }
        const auto encoded=std::chrono::steady_clock::now();
        encodeMs_=std::chrono::duration<double,std::milli>(encoded-prepared).count();
        gpu_->SubmitAndWait();gpuMs_=gpu_->LastGPUMs();joinedPrefillPairs_=gpu_->LastJoinedPrefillPairs();constantHeadDispatches_=gpu_->LastConstantHeadDispatches();singleCoderHeadDispatches_=gpu_->LastSingleCoderHeadDispatches();residualNormDispatches_=gpu_->LastResidualNormDispatches();packedPrefillNormDispatches_=gpu_->LastPackedPrefillNormDispatches();globalHalfAlphaDispatches_=gpu_->LastGlobalHalfAlphaDispatches();wideBatchPrefillDispatches_=gpu_->LastWideBatchPrefillDispatches();interleavedBodyDispatches_=gpu_->LastInterleavedBodyDispatches();coalescedBlockDispatches_=gpu_->LastCoalescedBlockDispatches();coalescedPrefillDispatches_=gpu_->LastCoalescedPrefillDispatches();mlpGraphDispatches_=gpu_->LastMlpGraphDispatches();joinedGlobalMlpBlocks_=gpu_->LastJoinedGlobalMlpBlocks();
        prefillPartialSubmissions_=gpu_->LastPartialSubmissions();
        screenedHeadDispatches_=gpu_->LastScreenedHeadDispatches();screenedHeadSeedRows_=gpu_->LastScreenedHeadSeedRows();interleavedScreenDispatches_=gpu_->LastInterleavedScreenDispatches();
        fusedHeadArgmaxDispatches_=gpu_->LastFusedHeadArgmaxDispatches();bodyShapeDispatches_=gpu_->LastBodyShapeDispatches();bodyShapeMisses_=gpu_->LastBodyShapeMisses();
        deferredZeroBytes_=gpu_->LastDeferredZeroBytes();deferredZeroStates_=gpu_->LastDeferredZeroStates();
        waitMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-encoded).count();
    }catch(...){screenedHeadDispatches_=0;screenedHeadSeedRows_=0;interleavedScreenDispatches_=0;prefillPartialSubmissions_=0;fusedHeadArgmaxDispatches_=0;bodyShapeDispatches_=bodyShapeMisses_=0;deferredZeroBytes_=deferredZeroStates_=0;kvInplaceUpdates_=0;joinedPrefillPairs_=0;constantHeadDispatches_=0;singleCoderHeadDispatches_=0;residualNormDispatches_=0;packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;wideBatchPrefillDispatches_=0;interleavedBodyDispatches_=0;coalescedPrefillDispatches_=0;coalescedBlockDispatches_=0;mlpGraphDispatches_=0;joinedGlobalMlpBlocks_=0;gpu_->Abort();throw;}
    hiddenShape_=inputShape[0];
    auto result=X::Value::List(Host());
    if(predictionToken>=0) {
        auto prediction=gpu_->Download(prediction_);
        for(size_t row=0;row<states_[0].rows;++row) {
            if(prediction[row*2+1]!=0)throw std::runtime_error("Nonfinite Metal session logits");
            result.Append(X::Value(int64_t(prediction[row*2])));
        }
    }
    inferenceMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    poisoned_=false;return result;
}
X::Value MetalDecoderSession::Snapshot(int64_t layer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||layer<0||uint64_t(layer)>=states_.size())throw std::invalid_argument("Metal session state snapshot invalid");
    auto result=X::Value::List(Host());const auto& state=states_[layer];
    for(size_t k=0;k<2;++k)result.Append(TensorValue(Host(),*gpu_,state.matrix[k],state.shape[k]));
    return result;
}
X::Value MetalDecoderSession::Hidden() {
    std::lock_guard<std::mutex> lock(mutex_);
    if(!gpu_||poisoned_||hiddenShape_.empty())throw std::logic_error("Metal session hidden unavailable");
    return TensorValue(Host(),*gpu_,hidden_,hiddenShape_);
}
}
