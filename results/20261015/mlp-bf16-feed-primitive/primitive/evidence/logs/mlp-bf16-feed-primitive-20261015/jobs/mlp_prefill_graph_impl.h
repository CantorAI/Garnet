// SPDX-License-Identifier: Apache-2.0
// Included inside namespace Garnet after the native matrix/context definitions.
bool MetalContext::EncodeMlpPrefillGraph(const MetalMatrix& input,
    const MetalBF16Weight& gateWeight,const MetalBF16Weight& upWeight,
    const MetalBF16Weight& downWeight,const MetalMatrix& output) {
#ifdef GARNET_METAL_MLP_PREFILL_GRAPH
    @autoreleasepool {
    if(@available(macOS 14.0,*)) {
    if(!impl_->command)throw std::logic_error("Begin a Metal command batch first");
    if(!input.impl_||!output.impl_||!gateWeight.impl_||!upWeight.impl_||!downWeight.impl_)return false;
    auto a=input.impl_->matrix,c=output.impl_->matrix;
    auto g=gateWeight.impl_,u=upWeight.impl_,d=downWeight.impl_;
    bool rows=a.rows>2&&a.rows<=128;
#ifdef GARNET_METAL_QWEN35_WIDE_BATCH_PREFILL
    rows=rows||a.rows==160;
#endif
    if(!rows||a.columns>1024||g->rows>8192||a.rowBytes!=a.columns*4||c.rowBytes!=c.columns*4)return false;
    if(a.device!=impl_->device||c.device!=impl_->device||g->buffer.device!=impl_->device||
        u->buffer.device!=impl_->device||d->buffer.device!=impl_->device)
        throw std::invalid_argument("MLP graph matrices belong to another device");
    if(g->columns!=a.columns||u->columns!=a.columns||g->rows!=u->rows||
        d->columns!=g->rows||d->rows!=a.columns||c.rows!=a.rows||c.columns!=a.columns)
        throw std::invalid_argument("MLP graph dependency shape mismatch");
    if(c.data==a.data||c.data==g->buffer||c.data==u->buffer||c.data==d->buffer)
        throw std::invalid_argument("Aliased MLP graph output");
    if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_.get())))
        throw std::logic_error("Uninitialized MLP graph input");
    if(output.impl_->pending&&!impl_->pendingOutputs.count(output.impl_.get()))
        throw std::logic_error("MLP graph output is in another command batch");
    const std::vector<uintptr_t> key={reinterpret_cast<uintptr_t>(input.impl_.get()),
        reinterpret_cast<uintptr_t>(output.impl_.get()),reinterpret_cast<uintptr_t>(g.get()),
        reinterpret_cast<uintptr_t>(u.get()),reinterpret_cast<uintptr_t>(d.get()),a.rows,a.columns,g->rows};
    auto found=impl_->mlpGraphs.find(key);
    if(found==impl_->mlpGraphs.end()) {
        // Bound retained programs/buffers for callers that continually create new slots.
        if(impl_->mlpGraphs.size()>=256)return false;
        for(auto w:{g,u,d})
            if(!w->buffer||w->rows>impl_->device.maxBufferLength/2/w->columns||
                w->buffer.length<w->rows*w->columns*2)return false;
        Impl::MlpGraph item;item.graph=[MPSGraph new];item.feeds=[NSMutableDictionary dictionary];item.results=[NSMutableDictionary dictionary];
        item.inputOwner=input;item.outputOwner=output;item.gateOwner=gateWeight;item.upOwner=upWeight;item.downOwner=downWeight;
        auto graph=item.graph;auto x=[graph placeholderWithShape:@[@(a.rows),@(a.columns)] dataType:MPSDataTypeFloat32 name:nil];
        item.feeds[x]=[[MPSGraphTensorData alloc] initWithMTLBuffer:a.data shape:@[@(a.rows),@(a.columns)] dataType:MPSDataTypeFloat32];
        MPSGraphTensor* weights[3];size_t index=0;
        for(auto w:{g,u,d}) {
            auto tensor=[graph placeholderWithShape:@[@(w->rows),@(w->columns)] dataType:MPSDataTypeBFloat16 name:nil];
            if(tensor.dataType!=MPSDataTypeBFloat16)throw std::runtime_error("Original BF16 feed dtype changed");
            item.feeds[tensor]=[[MPSGraphTensorData alloc] initWithMTLBuffer:w->buffer shape:@[@(w->rows),@(w->columns)] dataType:MPSDataTypeBFloat16];
            weights[index++]=[graph castTensor:tensor toType:MPSDataTypeFloat32 name:nil];
        }
        auto product=[&](MPSGraphTensor* lhs,MPSGraphTensor* rhs){return [graph matrixMultiplicationWithPrimaryTensor:lhs secondaryTensor:[graph transposeTensor:rhs dimension:0 withDimension:1 name:nil] name:nil];};
        auto gate=product(x,weights[0]),up=product(x,weights[1]);float oneValue=1.f;
        auto one=[graph constantWithData:[NSData dataWithBytes:&oneValue length:4] shape:@[@1] dataType:MPSDataTypeFloat32];
        auto neg=[graph negativeWithTensor:gate name:nil],exponential=[graph exponentWithTensor:neg name:nil];
        auto denominator=[graph additionWithPrimaryTensor:one secondaryTensor:exponential name:nil];
        auto silu=[graph divisionWithPrimaryTensor:gate secondaryTensor:denominator name:nil];
        auto activated=[graph multiplicationWithPrimaryTensor:silu secondaryTensor:up name:nil];
        auto result=product(activated,weights[2]);
        for(MPSGraphTensor* tensor in @[x,weights[0],weights[1],weights[2],gate,up,one,neg,exponential,denominator,silu,activated,result])
            if(tensor.dataType!=MPSDataTypeFloat32)throw std::runtime_error("MLP graph tensor dtype changed");
        item.results[result]=[[MPSGraphTensorData alloc] initWithMTLBuffer:c.data shape:@[@(c.rows),@(c.columns)] dataType:MPSDataTypeFloat32];
        found=impl_->mlpGraphs.emplace(key,std::move(item)).first;
    }
    auto completion=std::make_shared<Impl::MlpCompletion>();
    auto descriptor=[MPSGraphExecutionDescriptor new];descriptor.waitUntilCompleted=NO;
    descriptor.completionHandler=^(MPSGraphTensorDataDictionary*,NSError* error){
        std::lock_guard<std::mutex> lock(completion->mutex);
        if(error){const char* message=error.localizedDescription.UTF8String;completion->error=message?message:"MLP command-buffer graph failed";}
        completion->complete=true;completion->condition.notify_all();
    };
    output.impl_->ready=false;output.impl_->pending=true;
    if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
    auto buffer=[MPSCommandBuffer commandBufferWithCommandBuffer:impl_->command];
    if(!buffer)throw std::runtime_error("Cannot wrap native command buffer for MLP graph");
    auto original=buffer.rootCommandBuffer;
    impl_->mlpCompletions.push_back(completion);
    @try {
        auto& item=found->second;
        [item.graph encodeToCommandBuffer:buffer feeds:item.feeds targetOperations:nil resultsDictionary:item.results executionDescriptor:descriptor];
    } @catch(NSException* exception) {
        impl_->command=buffer.rootCommandBuffer;
        auto barrier=[impl_->queue commandBuffer];
        if(barrier){[barrier commit];[barrier waitUntilCompleted];}
        const char* message=exception.reason.UTF8String;throw std::runtime_error(message?message:"MLP command-buffer graph exception");
    }
    // Graph encoding is allowed to commitAndContinue. Every subsequent native
    // encoder must use its current root, never the possibly committed original.
    impl_->command=buffer.rootCommandBuffer;
    if(original!=impl_->command) {
        if(original.status<MTLCommandBufferStatusCommitted)
            throw std::runtime_error("MPS graph replaced an uncommitted root");
        if(!impl_->submitted)impl_->submitted=[NSMutableArray new];
        [impl_->submitted addObject:original];
    }
    ++impl_->mlpGraphDispatches;return true;
    } else return false;
    }
#else
    (void)input;(void)gateWeight;(void)upWeight;(void)downWeight;(void)output;return false;
#endif
}
