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
    hidden_=MetalMatrix();prediction_=MetalMatrix();hiddenShape_.clear();states_.clear();owners_.clear();head_=nullptr;
    gpu_.reset();poisoned_=false;inferenceMs_=0;prepareMs_=encodeMs_=gpuMs_=waitMs_=0;
}
void MetalDecoderSession::Close() {std::lock_guard<std::mutex> lock(mutex_);Clear();}
void MetalDecoderSession::ResetUnlocked() {
    if(!gpu_)throw std::logic_error("Metal session is closed");
    for(auto& state:states_)for(size_t i=0;i<2;++i)
        gpu_->Update(state.matrix[i],std::vector<float>(state.rows*state.columns[i],0));
    hidden_=MetalMatrix();hiddenShape_.clear();poisoned_=false;inferenceMs_=0;prepareMs_=encodeMs_=gpuMs_=waitMs_=0;
}
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
            }
            states_.push_back(std::move(state));owners_.push_back(std::move(owner));
        }
        if(head.IsValid()&&!head.IsNone()) {
            head_=Graph(head,Host());std::lock_guard<std::mutex> graphLock(head_->mutex_);
            if(!head_->ready_||head_->inputs_.size()!=1||head_->inputShapes_[0]!=std::vector<int64_t>{int64_t(batch),int64_t(width)})
                throw std::invalid_argument("Metal session vocabulary head dimensions mismatch");
            headGeneration_=head_->generation_;owners_.push_back(head);prediction_=gpu_->Allocate(batch,2);
        }
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
    if(hiddenShape_!=inputShape[0]) {
        const auto& slot=layers[0]->slots_[layers[0]->inputs_[0]];
        hidden_=gpu_->Allocate(slot.rows,slot.columns);
    }
    const auto prepared=std::chrono::steady_clock::now();
    prepareMs_=std::chrono::duration<double,std::milli>(prepared-start).count();
    poisoned_=true;gpu_->Begin();
    try {
        auto current=layers[0]->slots_[layers[0]->inputs_[0]].matrix;
        for(size_t i=0;i<layers.size();++i) {
            auto graph=layers[i];std::array<MetalMatrix,3> saved;
            for(size_t k=0;k<3;++k)saved[k]=graph->slots_[graph->inputs_[k]].matrix;
            graph->slots_[graph->inputs_[0]].matrix=current;
            graph->slots_[graph->inputs_[1]].matrix=states_[i].matrix[0];
            graph->slots_[graph->inputs_[2]].matrix=states_[i].matrix[1];
            try {graph->Encode(*gpu_,false);}
            catch(...){for(size_t k=0;k<3;++k)graph->slots_[graph->inputs_[k]].matrix=saved[k];throw;}
            for(size_t k=0;k<3;++k)graph->slots_[graph->inputs_[k]].matrix=saved[k];
            current=graph->slots_[graph->outputParts_[0]].matrix;
            // Keep the new state private to this session without copying it
            // back. The previous state becomes this graph's next output
            // storage; graph locks cover the exchange and GPU completion.
            for(size_t k=0;k<2;++k)if(!graph->SwapOutputStorage(k+1,states_[i].matrix[k]))
                gpu_->EncodeSliceGroups(graph->slots_[graph->outputParts_[k+1]].matrix,
                    states_[i].matrix[k],1,states_[i].columns[k],0,states_[i].columns[k]);
        }
        const size_t hiddenWidth=size_t(inputShape[0].back())*chunk;
        gpu_->EncodeSliceGroups(current,hidden_,1,hiddenWidth,0,hiddenWidth);
        if(predictionToken>=0) {
            auto& headInput=head_->slots_[head_->inputs_[0]];
            gpu_->EncodeSliceGroups(current,headInput.matrix,1,chunk*headInput.columns,
                size_t(predictionToken)*headInput.columns,headInput.columns);
            head_->Encode(*gpu_);
            gpu_->EncodeArgmax(head_->slots_[head_->output_].matrix,prediction_);
        }
        const auto encoded=std::chrono::steady_clock::now();
        encodeMs_=std::chrono::duration<double,std::milli>(encoded-prepared).count();
        gpu_->SubmitAndWait();gpuMs_=gpu_->LastGPUMs();
        waitMs_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-encoded).count();
    }catch(...){gpu_->Abort();throw;}
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
