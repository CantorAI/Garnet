// SPDX-License-Identifier: Apache-2.0
// Included inside Garnet after the Metal implementation types are complete.
bool MetalContext::PrepareScreenedHead(const MetalBF16Weight& weight) {
#ifndef GARNET_METAL_SCREENED_HEAD
    (void)weight;return false;
#else
    @autoreleasepool {
        if(!weight.impl_)return false;auto w=weight.impl_;
        if(w->buffer.device!=impl_->device||w->rows<=8192||w->rows>16777216||
            (w->columns!=896&&w->columns!=1024)||!w->buffer.contents||
            !impl_->kernels->screenPrepare||!impl_->kernels->screenProjection)return false;
        if(impl_->command)throw std::logic_error("Prepare screened weights before beginning a command batch");
        std::lock_guard<std::mutex> lock(w->screenMutex);
        if(w->screenAttempted)return w->screenIndex&&w->screenMetadata;
        w->screenAttempted=true;
        const size_t bytes=Elements(w->rows,w->columns),metadataBytes=Elements(w->rows,size_t(16));
        if(bytes>impl_->device.maxBufferLength||metadataBytes>impl_->device.maxBufferLength)return false;
        auto index=[impl_->device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        auto metadata=[impl_->device newBufferWithLength:metadataBytes options:MTLResourceStorageModeShared];
        if(!index||!metadata)return false;
        auto original=static_cast<const uint16_t*>(w->buffer.contents);auto encoded=static_cast<int8_t*>(index.contents);auto info=static_cast<float*>(metadata.contents);
        for(size_t row=0;row<w->rows;++row)if(!EncodeScreenedHeadRow(original+row*w->columns,w->columns,encoded+row*w->columns,info+row*4))return false;
        w->screenIndex=index;w->screenMetadata=metadata;return true;
    }
#endif
}
size_t MetalContext::LastScreenedHeadDispatches() const {return impl_->screenedHeadDispatches;}
size_t MetalContext::LastScreenedHeadSeedRows() const {return impl_->screenedHeadSeedRows;}
bool MetalContext::EncodeScreenedArgmax(const MetalMatrix& input,const MetalBF16Weight& weight,const MetalMatrix& output) {
#ifndef GARNET_METAL_SCREENED_HEAD
    (void)input;(void)weight;(void)output;return false;
#else
    @autoreleasepool {
        if(!input.impl_||!weight.impl_||!output.impl_)return false;
        auto a=input.impl_->matrix,o=output.impl_->matrix;auto w=weight.impl_;
        if(a.rows<1||a.rows>2||w->rows<=8192||w->rows>16777216||(w->columns!=896&&w->columns!=1024))return false;
        std::lock_guard<std::mutex> lock(w->screenMutex);
        if(!w->screenIndex||!w->screenMetadata)return false;
        auto prep=impl_->kernels->screenPrepare,projection=impl_->kernels->screenProjection,seedPipeline=impl_->kernels->bf16Decode;
        for(auto pipeline:{prep,projection,seedPipeline})if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128||
            pipeline.staticThreadgroupMemoryLength>impl_->device.maxThreadgroupMemoryLength)return false;
        if(!impl_->command)throw std::logic_error("Begin screened head command batch first");
        if(a.device!=impl_->device||o.device!=impl_->device||w->buffer.device!=impl_->device||
            a.columns!=w->columns||o.rows!=a.rows||o.columns!=2||a.data==o.data||o.data==w->buffer||
            o.data==w->screenIndex||o.data==w->screenMetadata)throw std::invalid_argument("Screened head shape/device/alias mismatch");
        if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_.get())))throw std::logic_error("Uninitialized screened head input");
        if(output.impl_->pending&&!impl_->pendingOutputs.count(output.impl_.get()))throw std::logic_error("Screened prediction belongs to another batch");
        const std::array<size_t,3> key{a.rows,w->rows,w->columns};auto found=impl_->screenScratch.find(key);
        if(found==impl_->screenScratch.end()){
            Impl::ScreenScratch item;item.seed=Allocate(a.rows,kScreenedHeadSeedRows);item.scores=Allocate(a.rows,w->rows);
            item.stats=[impl_->device newBufferWithLength:a.rows*(w->columns/128+2)*4 options:MTLResourceStorageModeShared];
            item.visits=[impl_->device newBufferWithLength:w->rows*8 options:MTLResourceStorageModeShared];
            if(!item.stats||!item.visits)throw std::runtime_error("Screened head scratch allocation failed");
            found=impl_->screenScratch.emplace(key,std::move(item)).first;
        }
        auto& scratch=found->second;auto seed=scratch.seed.impl_->matrix,scores=scratch.scores.impl_->matrix;
        if(seed.rowBytes/4!=kScreenedHeadSeedRows)return false;
        for(const auto& target:{scratch.seed.impl_,scratch.scores.impl_,output.impl_}){
            target->ready=false;target->pending=true;if(impl_->pendingOutputs.insert(target.get()).second)impl_->outputs.push_back(target);
        }
        struct Params{uint64_t rows,inner,columns,aStride,oStride;};
        Params sp{a.rows,a.columns,kScreenedHeadSeedRows,a.rowBytes/4,seed.rowBytes/4},p{a.rows,a.columns,w->rows,a.rowBytes/4,scores.rowBytes/4};
        auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Screen seed encoder unavailable");
        [encoder setComputePipelineState:seedPipeline];[encoder setBuffer:a.data offset:0 atIndex:0];[encoder setBuffer:w->buffer offset:0 atIndex:1];
        [encoder setBuffer:seed.data offset:0 atIndex:2];[encoder setBytes:&sp length:sizeof(sp) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(kScreenedHeadSeedRows/16,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
        encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Screen guard encoder unavailable");
        [encoder setComputePipelineState:prep];[encoder setBuffer:a.data offset:0 atIndex:0];[encoder setBuffer:seed.data offset:0 atIndex:1];
        [encoder setBuffer:scratch.stats offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(a.rows,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
        encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Screen refinement encoder unavailable");
        [encoder setComputePipelineState:projection];[encoder setBuffer:a.data offset:0 atIndex:0];[encoder setBuffer:w->buffer offset:0 atIndex:1];
        [encoder setBuffer:scores.data offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
        [encoder setBuffer:scratch.stats offset:0 atIndex:4];[encoder setBuffer:w->screenMetadata offset:0 atIndex:5];
        [encoder setBuffer:scratch.visits offset:0 atIndex:6];[encoder setBuffer:seed.data offset:0 atIndex:7];[encoder setBuffer:w->screenIndex offset:0 atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake((w->rows+15)/16,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
        EncodeArgmax(scratch.scores,output);
        impl_->screenDiagnosticKey=key;impl_->screenDiagnosticReady=true;++impl_->screenedHeadDispatches;impl_->screenedHeadSeedRows=sp.columns;return true;
    }
#endif
}
MetalScreenedDiagnostics MetalContext::ScreenedDiagnostics() const {
#ifndef GARNET_METAL_SCREENED_HEAD
    throw std::logic_error("Screened head experiment disabled");
#else
    if(impl_->command||!impl_->screenDiagnosticReady)throw std::logic_error("No completed screened diagnostic result");
    const auto& key=impl_->screenDiagnosticKey;const auto& item=impl_->screenScratch.at(key);
    MetalScreenedDiagnostics result;result.rows=key[0];result.columns=key[1];result.maskedLogits=Download(item.scores);
    auto visits=static_cast<const uint32_t*>(item.visits.contents);auto stats=static_cast<const float*>(item.stats.contents);
    result.kept.resize(result.rows*result.columns);result.querySafe.resize(result.rows);
    for(size_t row=0;row<result.rows;++row){result.querySafe[row]=stats[row*(key[2]/128+2)+2];
        for(size_t col=0;col<result.columns;++col)result.kept[row*result.columns+col]=(visits[result.columns+col]>>row)&1;
    }
    return result;
#endif
}
