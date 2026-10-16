"""Private actual-context prototype: prepare exact union weights and activation-only output."""
SHADER='''
struct PreparedUnionParams {ulong rows,columns,iStride,oStride;};
kernel void prepared_union_silu(device const float* joined [[buffer(0)]],device float* out [[buffer(1)]],constant PreparedUnionParams& p [[buffer(2)]],uint id [[thread_position_in_grid]]) {
    if(id>=p.rows*p.columns)return;
    ulong row=id/p.columns,col=id%p.columns;float gate=joined[row*p.iStride+col],up=joined[row*p.iStride+p.columns+col];
    out[row*p.oStride+col]=(gate/(1.0f+exp(-gate)))*up;
}
'''
METHODS='''
bool MetalContext::PrepareUnionMlp(const MetalBF16Weight& gate,const MetalBF16Weight& up,size_t rows) {
    @autoreleasepool {
    if(!gate.impl_||!up.impl_||gate.impl_==up.impl_)return false;
    auto first=gate.impl_,second=up.impl_;
    if((rows!=24&&rows!=44&&rows!=26&&rows!=52&&rows!=80&&rows!=160)||first->floatPrefill||second->floatPrefill||
        first->visionPrefill||second->visionPrefill||first->rows!=second->rows||first->columns!=second->columns||first->columns%8)return false;
    if(impl_->command)throw std::logic_error("Prepare union weights before Begin");
    if(first->buffer.device!=impl_->device||second->buffer.device!=impl_->device)throw std::invalid_argument("Union weight device mismatch");
    auto identity=std::make_pair(static_cast<const void*>(first.get()),static_cast<const void*>(second.get()));auto found=impl_->preparedUnionPairs.find(identity);
    if(found==impl_->preparedUnionPairs.end()) {
        if(impl_->preparedUnionPairs.size()>=128)return false;
        const size_t n=first->rows,k=first->columns,count=Elements(n,k),stride=[MPSMatrixDescriptor rowBytesFromColumns:k dataType:MPSDataTypeFloat16];
        if(stride>impl_->device.maxBufferLength/2/n)return false;
        uint16_t largest=0;for(auto w:{first,second})for(size_t i=0;i<count;++i){uint16_t b;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+i*2,2);largest=std::max(largest,uint16_t(b&32767));}
        int32_t scale=0;if(!ExactHalfRowScale(largest,scale))return false;
        auto buffer=[impl_->device newBufferWithLength:2*n*stride options:MTLResourceStorageModeShared];if(!buffer)throw std::runtime_error("Prepared union allocation failed");std::memset(buffer.contents,0,buffer.length);
        for(size_t part=0;part<2;++part){auto w=part?second:first;
            for(size_t row=0;row<n;++row)for(size_t col=0;col<k;++col){uint16_t b,h;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+2*(row*k+col),2);
                if(!EncodeExactScaledHalf(b,scale,h))return false;
                std::memcpy(static_cast<char*>(buffer.contents)+(part*n+row)*stride+col*2,&h,2);}
            for(size_t row=0;row<n;++row)for(size_t col=0;col<k;++col){uint16_t b,h;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+2*(row*k+col),2);std::memcpy(&h,static_cast<const char*>(buffer.contents)+(part*n+row)*stride+col*2,2);
                __fp16 half;std::memcpy(&half,&h,2);float value=std::ldexp(float(half),-scale);uint32_t bits;std::memcpy(&bits,&value,4);
                if(bits!=(uint32_t(b)<<16))throw std::runtime_error("Prepared union original-bit roundtrip failed");}
        }
        auto desc=[MPSMatrixDescriptor matrixDescriptorWithRows:2*n columns:k rowBytes:stride dataType:MPSDataTypeFloat16];auto view=[[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];if(!view)throw std::runtime_error("Prepared union view failed");
        Impl::PreparedUnionPair pair;pair.firstOwner=gate;pair.secondOwner=up;pair.weight=view;pair.scale=scale;
        found=impl_->preparedUnionPairs.emplace(identity,std::move(pair)).first;
    }
    auto& pair=found->second;if(!pair.shapes.count(rows)){
        Impl::PreparedUnionShape shape;shape.scratch=Allocate(rows,first->rows*2);
        shape.product=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:first->rows*2 interiorColumns:first->columns alpha:std::ldexp(1.0,-pair.scale) beta:0];
        if(!shape.product)throw std::runtime_error("Prepared union product failed");pair.shapes.emplace(rows,std::move(shape));
    }
    return true;
    }
}
bool MetalContext::EncodePreparedUnionMlp(const MetalMatrix& input,const MetalBF16Weight& gate,const MetalBF16Weight& up,const MetalMatrix& activated) {
    if(!input.impl_||!gate.impl_||!up.impl_||!activated.impl_)return false;
    auto identity=std::make_pair(static_cast<const void*>(gate.impl_.get()),static_cast<const void*>(up.impl_.get()));auto found=impl_->preparedUnionPairs.find(identity);
    if(found==impl_->preparedUnionPairs.end())return false;auto a=input.impl_->matrix,o=activated.impl_->matrix;auto& pair=found->second;auto shape=pair.shapes.find(a.rows);if(shape==pair.shapes.end())return false;
    if(!impl_->command)throw std::logic_error("Begin a union command first");
    if(a.device!=impl_->device||o.device!=impl_->device||a.columns!=gate.impl_->columns||o.columns!=gate.impl_->rows||o.rows!=a.rows||a.dataType!=MPSDataTypeFloat32||o.dataType!=MPSDataTypeFloat32||a.offset||o.offset||a.data==o.data)throw std::invalid_argument("Prepared union shape/device/alias mismatch");
    if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_.get())))throw std::logic_error("Uninitialized union input");
    if(activated.impl_->pending&&!impl_->pendingOutputs.count(activated.impl_.get()))throw std::logic_error("Union output belongs to another command batch");
    auto& item=shape->second;auto temporary=item.scratch.impl_->matrix;
    [item.product encodeToCommandBuffer:impl_->command leftMatrix:a rightMatrix:pair.weight resultMatrix:temporary];
    struct Params{uint64_t rows,columns,iStride,oStride;};Params p{a.rows,o.columns,temporary.rowBytes/4,o.rowBytes/4};
    auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Prepared union encoder failed");
    [encoder setComputePipelineState:impl_->kernels->preparedUnionSilu];[encoder setBuffer:temporary.data offset:0 atIndex:0];[encoder setBuffer:o.data offset:0 atIndex:1];[encoder setBytes:&p length:sizeof(p) atIndex:2];
    [encoder dispatchThreads:MTLSizeMake(Elements(o.rows,o.columns),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
    for(auto value:{item.scratch.impl_,activated.impl_}){value->ready=false;value->pending=true;if(impl_->pendingOutputs.insert(value.get()).second)impl_->outputs.push_back(value);}
    ++impl_->preparedUnionMlpDispatches;return true;
}
size_t MetalContext::LastPreparedUnionMlpDispatches() const {return impl_->preparedUnionMlpDispatches;}
size_t MetalContext::PreparedUnionWeightBytes() const {size_t bytes=0;for(const auto& entry:impl_->preparedUnionPairs)bytes+=entry.second.weight.data.length;return bytes;}
std::vector<float> MetalContext::DownloadPreparedUnion(const MetalBF16Weight& gate,const MetalBF16Weight& up,size_t rows) const {
    auto identity=std::make_pair(static_cast<const void*>(gate.impl_.get()),static_cast<const void*>(up.impl_.get()));auto found=impl_->preparedUnionPairs.find(identity);if(found==impl_->preparedUnionPairs.end())throw std::logic_error("Union pair not prepared");
    auto item=found->second.shapes.find(rows);if(item==found->second.shapes.end())throw std::logic_error("Union shape not prepared");return Download(item->second.scratch);
}
'''
def header(source):
    needle='    size_t LastJoinedGlobalMlpBlocks() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'''
    // Private prototype only: explicit preparation before Begin; original weights retained.
    bool PrepareUnionMlp(const MetalBF16Weight&,const MetalBF16Weight&,size_t rows);
    bool EncodePreparedUnionMlp(const MetalMatrix&,const MetalBF16Weight&,const MetalBF16Weight&,const MetalMatrix&);
    size_t LastPreparedUnionMlpDispatches() const;
    size_t PreparedUnionWeightBytes() const;
    std::vector<float> DownloadPreparedUnion(const MetalBF16Weight&,const MetalBF16Weight&,size_t rows) const;''')
def context(source):
    def replace(old,new,count=1):
        nonlocal source
        assert source.count(old)==count,(old,source.count(old));source=source.replace(old,new)
    replace('id<MTLComputePipelineState> silu;','id<MTLComputePipelineState> silu,preparedUnionSilu;')
    replace('    size_t joinedGlobalMlpBlocks=0;','''    size_t joinedGlobalMlpBlocks=0,preparedUnionMlpDispatches=0;
    struct PreparedUnionShape {MPSMatrixMultiplication* product=nil;MetalMatrix scratch;};
    struct PreparedUnionPair {MetalBF16Weight firstOwner,secondOwner;MPSMatrix* weight=nil;int32_t scale=0;std::map<size_t,PreparedUnionShape> shapes;};
    std::map<std::pair<const void*,const void*>,PreparedUnionPair> preparedUnionPairs;''')
    replace('kernel void add_matrix(',SHADER+'kernel void add_matrix(')
    old='impl_->kernels->silu=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"silu_mul"] error:&error];'
    replace(old,old+'''
        impl_->kernels->preparedUnionSilu=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"prepared_union_silu"] error:&error];
        if(!impl_->kernels->preparedUnionSilu||impl_->kernels->preparedUnionSilu.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Prepared union pipeline unavailable");''')
    replace('bool MetalContext::EncodeJoinedBF16Prefill(',METHODS+'\nbool MetalContext::EncodeJoinedBF16Prefill(')
    replace('impl_->joinedGlobalMlpBlocks=0;','impl_->joinedGlobalMlpBlocks=0;impl_->preparedUnionMlpDispatches=0;',2)
    return source
