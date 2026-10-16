"""Private context extension: exact heterogeneous Q/V with original K union; original weights untouched."""
SHADER='''
struct PreparedQvViewsOriginalKParams {ulong rows,qColumns,kColumns,vColumns,iStride,qStride,kStride,vStride;};
kernel void prepared_union_qv_views_scatter(device const float* joined [[buffer(0)]],device float* q [[buffer(1)]],device float* k [[buffer(2)]],device float* v [[buffer(3)]],constant PreparedQvViewsOriginalKParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    ulong columns=p.qColumns+p.vColumns;if(id>=p.rows*columns)return;
    ulong row=id/columns,col=id%columns;float value=joined[row*p.iStride+col];
    if(col<p.qColumns)q[row*p.qStride+col]=value;
    else v[row*p.vStride+col-p.qColumns]=value;
}
'''
METHODS='''
std::pair<MetalMatrix,MetalMatrix> MetalContext::PreparedUnionQvViewsOriginalKViews(const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,size_t rows)const{
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(q.impl_.get()),reinterpret_cast<uintptr_t>(k.impl_.get()),reinterpret_cast<uintptr_t>(v.impl_.get())};auto found=impl_->preparedQvViewsOriginalKGroups.find(identity);if(found==impl_->preparedQvViewsOriginalKGroups.end())throw std::logic_error("QV views not prepared");auto shape=found->second.shapes.find(rows);if(shape==found->second.shapes.end())throw std::logic_error("QV view shape not prepared");return {shape->second.qView,shape->second.vView};
}
bool MetalContext::PreparedUnionQvViewsOriginalKGuard(const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,size_t rows)const{
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(q.impl_.get()),reinterpret_cast<uintptr_t>(k.impl_.get()),reinterpret_cast<uintptr_t>(v.impl_.get())};auto& item=impl_->preparedQvViewsOriginalKGroups.at(identity).shapes.at(rows);if(!item.scratch.impl_->ready||item.scratch.impl_->pending)throw std::logic_error("QV view guard is pending");auto matrix=item.scratch.impl_->matrix;const auto* bytes=static_cast<const unsigned char*>(matrix.data.contents)+rows*matrix.rowBytes;for(size_t i=0;i<matrix.rowBytes;++i)if(bytes[i]!=0xA5)return false;return true;
}
bool MetalContext::PrepareUnionQvViewsOriginalK(const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,size_t rows) {
    @autoreleasepool {
    if(!q.impl_||!k.impl_||!v.impl_||q.impl_==k.impl_||q.impl_==v.impl_||k.impl_==v.impl_)return false;
    auto first=q.impl_,second=k.impl_,third=v.impl_;
    if((rows!=24&&rows!=44&&rows!=26&&rows!=52&&rows!=80&&rows!=160)||first->visionPrefill||second->visionPrefill||third->visionPrefill||first->columns!=second->columns||first->columns!=third->columns||first->columns%8)return false;
    if(impl_->command)throw std::logic_error("Prepare QV views with original K before Begin");
    for(auto w:{first,second,third})if(w->buffer.device!=impl_->device)throw std::invalid_argument("Prepared QV views with original K device mismatch");
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(first.get()),reinterpret_cast<uintptr_t>(second.get()),reinterpret_cast<uintptr_t>(third.get())};auto found=impl_->preparedQvViewsOriginalKGroups.find(identity);
    if(found==impl_->preparedQvViewsOriginalKGroups.end()){
        if(impl_->preparedQvViewsOriginalKGroups.size()>=64)return false;
        size_t columns=first->rows;for(auto w:{third}){if(w->rows>std::numeric_limits<size_t>::max()-columns)return false;columns+=w->rows;}
        const size_t inner=first->columns,stride=[MPSMatrixDescriptor rowBytesFromColumns:inner dataType:MPSDataTypeFloat16];
        if(stride>impl_->device.maxBufferLength/columns||stride>33554432/columns)return false;
        uint16_t largest=0;for(auto w:{first,second,third})for(size_t i=0;i<Elements(w->rows,inner);++i){uint16_t b;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+i*2,2);largest=std::max(largest,uint16_t(b&32767));}
        int32_t scale=0;if(!ExactHalfRowScale(largest,scale))return false;
        auto buffer=[impl_->device newBufferWithLength:columns*stride options:MTLResourceStorageModeShared];if(!buffer)throw std::runtime_error("Prepared QV views with original K allocation failed");std::memset(buffer.contents,0,buffer.length);size_t offset=0;
        // Preserve the previous triple admission and common scale. K participates
        // only in this conservative CPU eligibility check; its GPU projection stays original.
        for(size_t i=0;i<Elements(second->rows,inner);++i){uint16_t b,h;std::memcpy(&b,static_cast<const char*>(second->buffer.contents)+2*i,2);if(!EncodeExactScaledHalf(b,scale,h))return false;}
        for(auto w:{first,third}){
            for(size_t row=0;row<w->rows;++row)for(size_t col=0;col<inner;++col){uint16_t b,h;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+2*(row*inner+col),2);if(!EncodeExactScaledHalf(b,scale,h))return false;std::memcpy(static_cast<char*>(buffer.contents)+(offset+row)*stride+col*2,&h,2);}
            for(size_t row=0;row<w->rows;++row)for(size_t col=0;col<inner;++col){uint16_t b,h;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+2*(row*inner+col),2);std::memcpy(&h,static_cast<const char*>(buffer.contents)+(offset+row)*stride+col*2,2);__fp16 half;std::memcpy(&half,&h,2);float restored=std::ldexp(float(half),-scale);uint32_t bits;std::memcpy(&bits,&restored,4);if(bits!=(uint32_t(b)<<16))throw std::runtime_error("Prepared QV views with original K original-bit roundtrip failed");}
            offset+=w->rows;
        }
        auto desc=[MPSMatrixDescriptor matrixDescriptorWithRows:columns columns:inner rowBytes:stride dataType:MPSDataTypeFloat16];auto view=[[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];if(!view)throw std::runtime_error("Prepared QV views with original K view failed");
        Impl::PreparedQvViewsOriginalKGroup group;group.owners={q,k,v};group.weight=view;group.scale=scale;found=impl_->preparedQvViewsOriginalKGroups.emplace(identity,std::move(group)).first;
    }
    auto& group=found->second;if(!group.shapes.count(rows)){
        Impl::PreparedQvViewsOriginalKShape shape;shape.scratch=Allocate(rows+1,group.weight.rows);
        auto allocation=shape.scratch.impl_->matrix;auto shared=allocation.data;const size_t outputStride=allocation.rowBytes;
        std::memset(static_cast<char*>(shared.contents)+rows*outputStride,0xA5,outputStride);
        auto resultDescriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:group.weight.rows rowBytes:outputStride dataType:MPSDataTypeFloat32];
        shape.scratch.impl_->matrix=[[MPSMatrix alloc] initWithBuffer:shared descriptor:resultDescriptor];
        auto makeView=[&](size_t columns,size_t offset){auto state=std::make_shared<MetalMatrix::Impl>();auto desc=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns rowBytes:outputStride dataType:MPSDataTypeFloat32];state->matrix=[[MPSMatrix alloc] initWithBuffer:shared offset:offset descriptor:desc];if(!state->matrix)throw std::runtime_error("QV output view failed");return MetalMatrix(state);};
        shape.qView=makeView(first->rows,0);shape.vView=makeView(third->rows,first->rows*sizeof(float));shape.product=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:group.weight.rows interiorColumns:first->columns alpha:std::ldexp(1.,-group.scale) beta:0];if(!shape.product)throw std::runtime_error("Prepared QV views with original K product failed");group.shapes.emplace(rows,std::move(shape));
    }return true;
    }
}
bool MetalContext::EncodePreparedUnionQvViewsOriginalK(const MetalMatrix& input,const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,const MetalMatrix& qOut,const MetalMatrix& kOut,const MetalMatrix& vOut){
    if(!input.impl_||!q.impl_||!k.impl_||!v.impl_||!qOut.impl_||!kOut.impl_||!vOut.impl_)return false;
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(q.impl_.get()),reinterpret_cast<uintptr_t>(k.impl_.get()),reinterpret_cast<uintptr_t>(v.impl_.get())};auto found=impl_->preparedQvViewsOriginalKGroups.find(identity);if(found==impl_->preparedQvViewsOriginalKGroups.end())return false;
    auto a=input.impl_->matrix,qo=qOut.impl_->matrix,ko=kOut.impl_->matrix,vo=vOut.impl_->matrix;auto shape=found->second.shapes.find(a.rows);if(shape==found->second.shapes.end())return false;
    if(qOut.impl_!=shape->second.qView.impl_||vOut.impl_!=shape->second.vView.impl_||ko.offset)throw std::invalid_argument("Only canonical QV output views are writable");
    if(!impl_->command)throw std::logic_error("Begin a QV views with original K command first");
    if(a.device!=impl_->device||a.dataType!=MPSDataTypeFloat32||a.columns!=q.impl_->columns||a.offset)throw std::invalid_argument("Prepared QV views with original K input mismatch");
    if(qo.columns!=q.impl_->rows||ko.columns!=k.impl_->rows||vo.columns!=v.impl_->rows)throw std::invalid_argument("Prepared QV views with original K output columns mismatch");
    for(auto o:{qo,ko,vo})if(o.device!=impl_->device||o.rows!=a.rows||o.dataType!=MPSDataTypeFloat32||o.data==a.data)throw std::invalid_argument("Prepared QV views with original K output/input alias mismatch");
    if(qo.data==ko.data||ko.data==vo.data)throw std::invalid_argument("Prepared QV views with original K outputs alias");
    if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_.get())))throw std::logic_error("Uninitialized QV views with original K input");
    for(auto o:{qOut.impl_,kOut.impl_,vOut.impl_})if(o->pending&&!impl_->pendingOutputs.count(o.get()))throw std::logic_error("QV views with original K output belongs to another command batch");
    // All three outputs passed ownership/alias/readiness checks before any encode.
    // K retains its original EncodeBF16MatVec route, including split FP32 key accumulation.
    EncodeBF16MatVec(input,k,kOut);
    auto& item=shape->second;auto joined=item.scratch.impl_->matrix;[item.product encodeToCommandBuffer:impl_->command leftMatrix:a rightMatrix:found->second.weight resultMatrix:joined];
    // No scatter encode: Q and V are disjoint canonical views of this result.
    for(auto o:{item.scratch.impl_,qOut.impl_,kOut.impl_,vOut.impl_}){o->ready=false;o->pending=true;if(impl_->pendingOutputs.insert(o.get()).second)impl_->outputs.push_back(o);}++impl_->preparedQvViewsOriginalKDispatches;return true;
}
size_t MetalContext::LastPreparedUnionQvViewsOriginalKDispatches()const{return impl_->preparedQvViewsOriginalKDispatches;}
size_t MetalContext::PreparedUnionQvViewsOriginalKWeightBytes()const{size_t bytes=0;for(const auto& entry:impl_->preparedQvViewsOriginalKGroups)bytes+=entry.second.weight.data.length;return bytes;}
std::vector<float> MetalContext::DownloadPreparedUnionQvViewsOriginalK(const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,size_t rows)const{
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(q.impl_.get()),reinterpret_cast<uintptr_t>(k.impl_.get()),reinterpret_cast<uintptr_t>(v.impl_.get())};auto found=impl_->preparedQvViewsOriginalKGroups.find(identity);if(found==impl_->preparedQvViewsOriginalKGroups.end())throw std::logic_error("QV views with original K not prepared");auto item=found->second.shapes.find(rows);if(item==found->second.shapes.end())throw std::logic_error("QV views with original K shape not prepared");return Download(item->second.scratch);
}
'''
def header(source):
    needle='    size_t LastJoinedGlobalMlpBlocks() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'''
    // Private component output-view experiment. Only Download honors view offsets here;
    // downstream GPU consumers/host writes require independent integration and validation.
    std::pair<MetalMatrix,MetalMatrix> PreparedUnionQvViewsOriginalKViews(const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,size_t)const;
    bool PreparedUnionQvViewsOriginalKGuard(const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,size_t)const;
    bool PrepareUnionQvViewsOriginalK(const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,size_t);
    bool EncodePreparedUnionQvViewsOriginalK(const MetalMatrix&,const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&);
    size_t LastPreparedUnionQvViewsOriginalKDispatches()const;
    size_t PreparedUnionQvViewsOriginalKWeightBytes()const;
    std::vector<float> DownloadPreparedUnionQvViewsOriginalK(const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,size_t)const;''')
def context(source):
    def replace(old,new,count=1):
        nonlocal source
        assert source.count(old)==count,(old,source.count(old));source=source.replace(old,new)
    replace('id<MTLComputePipelineState> silu;','id<MTLComputePipelineState> silu,preparedQvViewsOriginalKScatter;')
    replace('    size_t joinedGlobalMlpBlocks=0;','''    size_t joinedGlobalMlpBlocks=0,preparedQvViewsOriginalKDispatches=0;
    struct PreparedQvViewsOriginalKShape{MPSMatrixMultiplication* product=nil;MetalMatrix scratch,qView,vView;};
    struct PreparedQvViewsOriginalKGroup{std::array<MetalBF16Weight,3> owners;MPSMatrix* weight=nil;int32_t scale=0;std::map<size_t,PreparedQvViewsOriginalKShape> shapes;};
    std::map<std::array<uintptr_t,3>,PreparedQvViewsOriginalKGroup> preparedQvViewsOriginalKGroups;''')
    replace('kernel void add_matrix(',SHADER+'kernel void add_matrix(')
    needle='impl_->kernels->silu=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"silu_mul"] error:&error];'
    replace(needle,needle+'''
        impl_->kernels->preparedQvViewsOriginalKScatter=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"prepared_union_qv_views_scatter"] error:&error];
        if(!impl_->kernels->preparedQvViewsOriginalKScatter||impl_->kernels->preparedQvViewsOriginalKScatter.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Prepared QV views with original K scatter pipeline unavailable");''')
    replace('bool MetalContext::EncodeJoinedBF16Prefill(',METHODS+'\nbool MetalContext::EncodeJoinedBF16Prefill(')
    replace('impl_->joinedGlobalMlpBlocks=0;','impl_->joinedGlobalMlpBlocks=0;impl_->preparedQvViewsOriginalKDispatches=0;',2)
    replace('static_cast<char*>(matrix.data.contents)+i*matrix.rowBytes','static_cast<char*>(matrix.data.contents)+matrix.offset+i*matrix.rowBytes')
    return source
