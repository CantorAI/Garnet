"""Private context extension: exact heterogeneous Q/V with original K union; original weights untouched."""
SHADER='''
struct PreparedQvOriginalKParams {ulong rows,qColumns,kColumns,vColumns,iStride,qStride,kStride,vStride;};
kernel void prepared_union_qv_scatter(device const float* joined [[buffer(0)]],device float* q [[buffer(1)]],device float* k [[buffer(2)]],device float* v [[buffer(3)]],constant PreparedQvOriginalKParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    ulong columns=p.qColumns+p.vColumns;if(id>=p.rows*columns)return;
    ulong row=id/columns,col=id%columns;float value=joined[row*p.iStride+col];
    if(col<p.qColumns)q[row*p.qStride+col]=value;
    else v[row*p.vStride+col-p.qColumns]=value;
}
'''
METHODS='''
bool MetalContext::PrepareUnionQvOriginalK(const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,size_t rows) {
    @autoreleasepool {
    if(!q.impl_||!k.impl_||!v.impl_||q.impl_==k.impl_||q.impl_==v.impl_||k.impl_==v.impl_)return false;
    auto first=q.impl_,second=k.impl_,third=v.impl_;
    if((rows!=24&&rows!=44&&rows!=26&&rows!=52&&rows!=80&&rows!=160)||first->visionPrefill||second->visionPrefill||third->visionPrefill||first->columns!=second->columns||first->columns!=third->columns||first->columns%8)return false;
    if(impl_->command)throw std::logic_error("Prepare QV with original K before Begin");
    for(auto w:{first,second,third})if(w->buffer.device!=impl_->device)throw std::invalid_argument("Prepared QV with original K device mismatch");
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(first.get()),reinterpret_cast<uintptr_t>(second.get()),reinterpret_cast<uintptr_t>(third.get())};auto found=impl_->preparedQvOriginalKGroups.find(identity);
    if(found==impl_->preparedQvOriginalKGroups.end()){
        if(impl_->preparedQvOriginalKGroups.size()>=64)return false;
        size_t columns=first->rows;for(auto w:{third}){if(w->rows>std::numeric_limits<size_t>::max()-columns)return false;columns+=w->rows;}
        const size_t inner=first->columns,stride=[MPSMatrixDescriptor rowBytesFromColumns:inner dataType:MPSDataTypeFloat16];
        if(stride>impl_->device.maxBufferLength/columns||stride>33554432/columns)return false;
        uint16_t largest=0;for(auto w:{first,second,third})for(size_t i=0;i<Elements(w->rows,inner);++i){uint16_t b;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+i*2,2);largest=std::max(largest,uint16_t(b&32767));}
        int32_t scale=0;if(!ExactHalfRowScale(largest,scale))return false;
        auto buffer=[impl_->device newBufferWithLength:columns*stride options:MTLResourceStorageModeShared];if(!buffer)throw std::runtime_error("Prepared QV with original K allocation failed");std::memset(buffer.contents,0,buffer.length);size_t offset=0;
        // Preserve the previous triple admission and common scale. K participates
        // only in this conservative CPU eligibility check; its GPU projection stays original.
        for(size_t i=0;i<Elements(second->rows,inner);++i){uint16_t b,h;std::memcpy(&b,static_cast<const char*>(second->buffer.contents)+2*i,2);if(!EncodeExactScaledHalf(b,scale,h))return false;}
        for(auto w:{first,third}){
            for(size_t row=0;row<w->rows;++row)for(size_t col=0;col<inner;++col){uint16_t b,h;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+2*(row*inner+col),2);if(!EncodeExactScaledHalf(b,scale,h))return false;std::memcpy(static_cast<char*>(buffer.contents)+(offset+row)*stride+col*2,&h,2);}
            for(size_t row=0;row<w->rows;++row)for(size_t col=0;col<inner;++col){uint16_t b,h;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+2*(row*inner+col),2);std::memcpy(&h,static_cast<const char*>(buffer.contents)+(offset+row)*stride+col*2,2);__fp16 half;std::memcpy(&half,&h,2);float restored=std::ldexp(float(half),-scale);uint32_t bits;std::memcpy(&bits,&restored,4);if(bits!=(uint32_t(b)<<16))throw std::runtime_error("Prepared QV with original K original-bit roundtrip failed");}
            offset+=w->rows;
        }
        auto desc=[MPSMatrixDescriptor matrixDescriptorWithRows:columns columns:inner rowBytes:stride dataType:MPSDataTypeFloat16];auto view=[[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];if(!view)throw std::runtime_error("Prepared QV with original K view failed");
        Impl::PreparedQvOriginalKGroup group;group.owners={q,k,v};group.weight=view;group.scale=scale;found=impl_->preparedQvOriginalKGroups.emplace(identity,std::move(group)).first;
    }
    auto& group=found->second;if(!group.shapes.count(rows)){
        Impl::PreparedQvOriginalKShape shape;shape.scratch=Allocate(rows,group.weight.rows);shape.product=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:group.weight.rows interiorColumns:first->columns alpha:std::ldexp(1.,-group.scale) beta:0];if(!shape.product)throw std::runtime_error("Prepared QV with original K product failed");group.shapes.emplace(rows,std::move(shape));
    }return true;
    }
}
bool MetalContext::EncodePreparedUnionQvOriginalK(const MetalMatrix& input,const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,const MetalMatrix& qOut,const MetalMatrix& kOut,const MetalMatrix& vOut){
    if(!input.impl_||!q.impl_||!k.impl_||!v.impl_||!qOut.impl_||!kOut.impl_||!vOut.impl_)return false;
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(q.impl_.get()),reinterpret_cast<uintptr_t>(k.impl_.get()),reinterpret_cast<uintptr_t>(v.impl_.get())};auto found=impl_->preparedQvOriginalKGroups.find(identity);if(found==impl_->preparedQvOriginalKGroups.end())return false;
    auto a=input.impl_->matrix,qo=qOut.impl_->matrix,ko=kOut.impl_->matrix,vo=vOut.impl_->matrix;auto shape=found->second.shapes.find(a.rows);if(shape==found->second.shapes.end())return false;
    if(!impl_->command)throw std::logic_error("Begin a QV with original K command first");
    if(a.device!=impl_->device||a.dataType!=MPSDataTypeFloat32||a.columns!=q.impl_->columns||a.offset)throw std::invalid_argument("Prepared QV with original K input mismatch");
    if(qo.columns!=q.impl_->rows||ko.columns!=k.impl_->rows||vo.columns!=v.impl_->rows)throw std::invalid_argument("Prepared QV with original K output columns mismatch");
    for(auto o:{qo,ko,vo})if(o.device!=impl_->device||o.rows!=a.rows||o.dataType!=MPSDataTypeFloat32||o.offset||o.data==a.data)throw std::invalid_argument("Prepared QV with original K output/input alias mismatch");
    if(qo.data==ko.data||qo.data==vo.data||ko.data==vo.data)throw std::invalid_argument("Prepared QV with original K outputs alias");
    if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_.get())))throw std::logic_error("Uninitialized QV with original K input");
    for(auto o:{qOut.impl_,kOut.impl_,vOut.impl_})if(o->pending&&!impl_->pendingOutputs.count(o.get()))throw std::logic_error("QV with original K output belongs to another command batch");
    // All three outputs passed ownership/alias/readiness checks before any encode.
    // K retains the original float-prefill descriptor, storage and MPS call.
    EncodeBF16MatVec(input,k,kOut);
    auto& item=shape->second;auto joined=item.scratch.impl_->matrix;[item.product encodeToCommandBuffer:impl_->command leftMatrix:a rightMatrix:found->second.weight resultMatrix:joined];
    struct Params{uint64_t rows,qColumns,kColumns,vColumns,iStride,qStride,kStride,vStride;};Params p{a.rows,qo.columns,ko.columns,vo.columns,joined.rowBytes/4,qo.rowBytes/4,ko.rowBytes/4,vo.rowBytes/4};auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Prepared QV with original K scatter encoder failed");
    [encoder setComputePipelineState:impl_->kernels->preparedQvOriginalKScatter];[encoder setBuffer:joined.data offset:0 atIndex:0];[encoder setBuffer:qo.data offset:0 atIndex:1];[encoder setBuffer:ko.data offset:0 atIndex:2];[encoder setBuffer:vo.data offset:0 atIndex:3];[encoder setBytes:&p length:sizeof(p) atIndex:4];[encoder dispatchThreads:MTLSizeMake(Elements(a.rows,joined.columns),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
    for(auto o:{item.scratch.impl_,qOut.impl_,kOut.impl_,vOut.impl_}){o->ready=false;o->pending=true;if(impl_->pendingOutputs.insert(o.get()).second)impl_->outputs.push_back(o);}++impl_->preparedQvOriginalKDispatches;return true;
}
size_t MetalContext::LastPreparedUnionQvOriginalKDispatches()const{return impl_->preparedQvOriginalKDispatches;}
size_t MetalContext::PreparedUnionQvOriginalKWeightBytes()const{size_t bytes=0;for(const auto& entry:impl_->preparedQvOriginalKGroups)bytes+=entry.second.weight.data.length;return bytes;}
std::vector<float> MetalContext::DownloadPreparedUnionQvOriginalK(const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,size_t rows)const{
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(q.impl_.get()),reinterpret_cast<uintptr_t>(k.impl_.get()),reinterpret_cast<uintptr_t>(v.impl_.get())};auto found=impl_->preparedQvOriginalKGroups.find(identity);if(found==impl_->preparedQvOriginalKGroups.end())throw std::logic_error("QV with original K not prepared");auto item=found->second.shapes.find(rows);if(item==found->second.shapes.end())throw std::logic_error("QV with original K shape not prepared");return Download(item->second.scratch);
}
'''
def header(source):
    needle='    size_t LastJoinedGlobalMlpBlocks() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'''
    // Private prepared exact heterogeneous QV with original K experiment; independent originals retained.
    bool PrepareUnionQvOriginalK(const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,size_t);
    bool EncodePreparedUnionQvOriginalK(const MetalMatrix&,const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&);
    size_t LastPreparedUnionQvOriginalKDispatches()const;
    size_t PreparedUnionQvOriginalKWeightBytes()const;
    std::vector<float> DownloadPreparedUnionQvOriginalK(const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,size_t)const;''')
def context(source):
    def replace(old,new,count=1):
        nonlocal source
        assert source.count(old)==count,(old,source.count(old));source=source.replace(old,new)
    replace('id<MTLComputePipelineState> silu;','id<MTLComputePipelineState> silu,preparedQvOriginalKScatter;')
    replace('    size_t joinedGlobalMlpBlocks=0;','''    size_t joinedGlobalMlpBlocks=0,preparedQvOriginalKDispatches=0;
    struct PreparedQvOriginalKShape{MPSMatrixMultiplication* product=nil;MetalMatrix scratch;};
    struct PreparedQvOriginalKGroup{std::array<MetalBF16Weight,3> owners;MPSMatrix* weight=nil;int32_t scale=0;std::map<size_t,PreparedQvOriginalKShape> shapes;};
    std::map<std::array<uintptr_t,3>,PreparedQvOriginalKGroup> preparedQvOriginalKGroups;''')
    replace('kernel void add_matrix(',SHADER+'kernel void add_matrix(')
    needle='impl_->kernels->silu=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"silu_mul"] error:&error];'
    replace(needle,needle+'''
        impl_->kernels->preparedQvOriginalKScatter=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"prepared_union_qv_scatter"] error:&error];
        if(!impl_->kernels->preparedQvOriginalKScatter||impl_->kernels->preparedQvOriginalKScatter.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Prepared QV with original K scatter pipeline unavailable");''')
    replace('bool MetalContext::EncodeJoinedBF16Prefill(',METHODS+'\nbool MetalContext::EncodeJoinedBF16Prefill(')
    replace('impl_->joinedGlobalMlpBlocks=0;','impl_->joinedGlobalMlpBlocks=0;impl_->preparedQvOriginalKDispatches=0;',2)
    return source
