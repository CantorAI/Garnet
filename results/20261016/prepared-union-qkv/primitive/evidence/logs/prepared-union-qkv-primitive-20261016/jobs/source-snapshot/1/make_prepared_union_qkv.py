"""Private context extension: exact heterogeneous Q/K/V union; original weights untouched."""
SHADER='''
struct PreparedQkvParams {ulong rows,qColumns,kColumns,vColumns,iStride,qStride,kStride,vStride;};
kernel void prepared_union_qkv_scatter(device const float* joined [[buffer(0)]],device float* q [[buffer(1)]],device float* k [[buffer(2)]],device float* v [[buffer(3)]],constant PreparedQkvParams& p [[buffer(4)]],uint id [[thread_position_in_grid]]) {
    ulong columns=p.qColumns+p.kColumns+p.vColumns;if(id>=p.rows*columns)return;
    ulong row=id/columns,col=id%columns;float value=joined[row*p.iStride+col];
    if(col<p.qColumns)q[row*p.qStride+col]=value;
    else if(col<p.qColumns+p.kColumns)k[row*p.kStride+col-p.qColumns]=value;
    else v[row*p.vStride+col-p.qColumns-p.kColumns]=value;
}
'''
METHODS='''
bool MetalContext::PrepareUnionQkv(const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,size_t rows) {
    @autoreleasepool {
    if(!q.impl_||!k.impl_||!v.impl_||q.impl_==k.impl_||q.impl_==v.impl_||k.impl_==v.impl_)return false;
    auto first=q.impl_,second=k.impl_,third=v.impl_;
    if((rows!=24&&rows!=44&&rows!=26&&rows!=52&&rows!=80&&rows!=160)||first->visionPrefill||second->visionPrefill||third->visionPrefill||first->columns!=second->columns||first->columns!=third->columns||first->columns%8)return false;
    if(impl_->command)throw std::logic_error("Prepare QKV before Begin");
    for(auto w:{first,second,third})if(w->buffer.device!=impl_->device)throw std::invalid_argument("Prepared QKV device mismatch");
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(first.get()),reinterpret_cast<uintptr_t>(second.get()),reinterpret_cast<uintptr_t>(third.get())};auto found=impl_->preparedQkvGroups.find(identity);
    if(found==impl_->preparedQkvGroups.end()){
        if(impl_->preparedQkvGroups.size()>=64)return false;
        size_t columns=first->rows;for(auto w:{second,third}){if(w->rows>std::numeric_limits<size_t>::max()-columns)return false;columns+=w->rows;}
        const size_t inner=first->columns,stride=[MPSMatrixDescriptor rowBytesFromColumns:inner dataType:MPSDataTypeFloat16];
        if(stride>impl_->device.maxBufferLength/columns||stride>33554432/columns)return false;
        uint16_t largest=0;for(auto w:{first,second,third})for(size_t i=0;i<Elements(w->rows,inner);++i){uint16_t b;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+i*2,2);largest=std::max(largest,uint16_t(b&32767));}
        int32_t scale=0;if(!ExactHalfRowScale(largest,scale))return false;
        auto buffer=[impl_->device newBufferWithLength:columns*stride options:MTLResourceStorageModeShared];if(!buffer)throw std::runtime_error("Prepared QKV allocation failed");std::memset(buffer.contents,0,buffer.length);size_t offset=0;
        for(auto w:{first,second,third}){
            for(size_t row=0;row<w->rows;++row)for(size_t col=0;col<inner;++col){uint16_t b,h;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+2*(row*inner+col),2);if(!EncodeExactScaledHalf(b,scale,h))return false;std::memcpy(static_cast<char*>(buffer.contents)+(offset+row)*stride+col*2,&h,2);}
            for(size_t row=0;row<w->rows;++row)for(size_t col=0;col<inner;++col){uint16_t b,h;std::memcpy(&b,static_cast<const char*>(w->buffer.contents)+2*(row*inner+col),2);std::memcpy(&h,static_cast<const char*>(buffer.contents)+(offset+row)*stride+col*2,2);__fp16 half;std::memcpy(&half,&h,2);float restored=std::ldexp(float(half),-scale);uint32_t bits;std::memcpy(&bits,&restored,4);if(bits!=(uint32_t(b)<<16))throw std::runtime_error("Prepared QKV original-bit roundtrip failed");}
            offset+=w->rows;
        }
        auto desc=[MPSMatrixDescriptor matrixDescriptorWithRows:columns columns:inner rowBytes:stride dataType:MPSDataTypeFloat16];auto view=[[MPSMatrix alloc] initWithBuffer:buffer descriptor:desc];if(!view)throw std::runtime_error("Prepared QKV view failed");
        Impl::PreparedQkvGroup group;group.owners={q,k,v};group.weight=view;group.scale=scale;found=impl_->preparedQkvGroups.emplace(identity,std::move(group)).first;
    }
    auto& group=found->second;if(!group.shapes.count(rows)){
        Impl::PreparedQkvShape shape;shape.scratch=Allocate(rows,group.weight.rows);shape.product=[[MPSMatrixMultiplication alloc] initWithDevice:impl_->device transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:group.weight.rows interiorColumns:first->columns alpha:std::ldexp(1.,-group.scale) beta:0];if(!shape.product)throw std::runtime_error("Prepared QKV product failed");group.shapes.emplace(rows,std::move(shape));
    }return true;
    }
}
bool MetalContext::EncodePreparedUnionQkv(const MetalMatrix& input,const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,const MetalMatrix& qOut,const MetalMatrix& kOut,const MetalMatrix& vOut){
    if(!input.impl_||!q.impl_||!k.impl_||!v.impl_||!qOut.impl_||!kOut.impl_||!vOut.impl_)return false;
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(q.impl_.get()),reinterpret_cast<uintptr_t>(k.impl_.get()),reinterpret_cast<uintptr_t>(v.impl_.get())};auto found=impl_->preparedQkvGroups.find(identity);if(found==impl_->preparedQkvGroups.end())return false;
    auto a=input.impl_->matrix,qo=qOut.impl_->matrix,ko=kOut.impl_->matrix,vo=vOut.impl_->matrix;auto shape=found->second.shapes.find(a.rows);if(shape==found->second.shapes.end())return false;
    if(!impl_->command)throw std::logic_error("Begin a QKV command first");
    if(a.device!=impl_->device||a.dataType!=MPSDataTypeFloat32||a.columns!=q.impl_->columns||a.offset)throw std::invalid_argument("Prepared QKV input mismatch");
    if(qo.columns!=q.impl_->rows||ko.columns!=k.impl_->rows||vo.columns!=v.impl_->rows)throw std::invalid_argument("Prepared QKV output columns mismatch");
    for(auto o:{qo,ko,vo})if(o.device!=impl_->device||o.rows!=a.rows||o.dataType!=MPSDataTypeFloat32||o.offset||o.data==a.data)throw std::invalid_argument("Prepared QKV output/input alias mismatch");
    if(qo.data==ko.data||qo.data==vo.data||ko.data==vo.data)throw std::invalid_argument("Prepared QKV outputs alias");
    if(!input.impl_->ready&&!(input.impl_->pending&&impl_->pendingOutputs.count(input.impl_.get())))throw std::logic_error("Uninitialized QKV input");
    for(auto o:{qOut.impl_,kOut.impl_,vOut.impl_})if(o->pending&&!impl_->pendingOutputs.count(o.get()))throw std::logic_error("QKV output belongs to another command batch");
    auto& item=shape->second;auto joined=item.scratch.impl_->matrix;[item.product encodeToCommandBuffer:impl_->command leftMatrix:a rightMatrix:found->second.weight resultMatrix:joined];
    struct Params{uint64_t rows,qColumns,kColumns,vColumns,iStride,qStride,kStride,vStride;};Params p{a.rows,qo.columns,ko.columns,vo.columns,joined.rowBytes/4,qo.rowBytes/4,ko.rowBytes/4,vo.rowBytes/4};auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Prepared QKV scatter encoder failed");
    [encoder setComputePipelineState:impl_->kernels->preparedQkvScatter];[encoder setBuffer:joined.data offset:0 atIndex:0];[encoder setBuffer:qo.data offset:0 atIndex:1];[encoder setBuffer:ko.data offset:0 atIndex:2];[encoder setBuffer:vo.data offset:0 atIndex:3];[encoder setBytes:&p length:sizeof(p) atIndex:4];[encoder dispatchThreads:MTLSizeMake(Elements(a.rows,joined.columns),1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[encoder endEncoding];
    for(auto o:{item.scratch.impl_,qOut.impl_,kOut.impl_,vOut.impl_}){o->ready=false;o->pending=true;if(impl_->pendingOutputs.insert(o.get()).second)impl_->outputs.push_back(o);}++impl_->preparedQkvDispatches;return true;
}
size_t MetalContext::LastPreparedUnionQkvDispatches()const{return impl_->preparedQkvDispatches;}
size_t MetalContext::PreparedUnionQkvWeightBytes()const{size_t bytes=0;for(const auto& entry:impl_->preparedQkvGroups)bytes+=entry.second.weight.data.length;return bytes;}
std::vector<float> MetalContext::DownloadPreparedUnionQkv(const MetalBF16Weight& q,const MetalBF16Weight& k,const MetalBF16Weight& v,size_t rows)const{
    const std::array<uintptr_t,3> identity{reinterpret_cast<uintptr_t>(q.impl_.get()),reinterpret_cast<uintptr_t>(k.impl_.get()),reinterpret_cast<uintptr_t>(v.impl_.get())};auto found=impl_->preparedQkvGroups.find(identity);if(found==impl_->preparedQkvGroups.end())throw std::logic_error("QKV not prepared");auto item=found->second.shapes.find(rows);if(item==found->second.shapes.end())throw std::logic_error("QKV shape not prepared");return Download(item->second.scratch);
}
'''
def header(source):
    needle='    size_t LastJoinedGlobalMlpBlocks() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'''
    // Private prepared exact heterogeneous QKV experiment; independent originals retained.
    bool PrepareUnionQkv(const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,size_t);
    bool EncodePreparedUnionQkv(const MetalMatrix&,const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&);
    size_t LastPreparedUnionQkvDispatches()const;
    size_t PreparedUnionQkvWeightBytes()const;
    std::vector<float> DownloadPreparedUnionQkv(const MetalBF16Weight&,const MetalBF16Weight&,const MetalBF16Weight&,size_t)const;''')
def context(source):
    def replace(old,new,count=1):
        nonlocal source
        assert source.count(old)==count,(old,source.count(old));source=source.replace(old,new)
    replace('id<MTLComputePipelineState> silu;','id<MTLComputePipelineState> silu,preparedQkvScatter;')
    replace('    size_t joinedGlobalMlpBlocks=0;','''    size_t joinedGlobalMlpBlocks=0,preparedQkvDispatches=0;
    struct PreparedQkvShape{MPSMatrixMultiplication* product=nil;MetalMatrix scratch;};
    struct PreparedQkvGroup{std::array<MetalBF16Weight,3> owners;MPSMatrix* weight=nil;int32_t scale=0;std::map<size_t,PreparedQkvShape> shapes;};
    std::map<std::array<uintptr_t,3>,PreparedQkvGroup> preparedQkvGroups;''')
    replace('kernel void add_matrix(',SHADER+'kernel void add_matrix(')
    needle='impl_->kernels->silu=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"silu_mul"] error:&error];'
    replace(needle,needle+'''
        impl_->kernels->preparedQkvScatter=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"prepared_union_qkv_scatter"] error:&error];
        if(!impl_->kernels->preparedQkvScatter||impl_->kernels->preparedQkvScatter.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Prepared QKV scatter pipeline unavailable");''')
    replace('bool MetalContext::EncodeJoinedBF16Prefill(',METHODS+'\nbool MetalContext::EncodeJoinedBF16Prefill(')
    replace('impl_->joinedGlobalMlpBlocks=0;','impl_->joinedGlobalMlpBlocks=0;impl_->preparedQkvDispatches=0;',2)
    return source
