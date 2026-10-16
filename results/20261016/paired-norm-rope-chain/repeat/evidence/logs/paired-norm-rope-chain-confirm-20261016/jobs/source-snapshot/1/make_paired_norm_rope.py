"""Private paired Q/K norm-RoPE dispatch; preserve the original per-head arithmetic body."""
METHODS='''
bool MetalContext::EncodePairedNormRope(const MetalMatrix& query,const MetalMatrix& key,
    const MetalMatrix& queryWeight,const MetalMatrix& keyWeight,const MetalMatrix& positions,const MetalMatrix& frequencies,
    const MetalMatrix& queryOut,const MetalMatrix& keyOut,size_t queryHeads,size_t keyHeads,size_t dim,size_t rotary,float epsilon,float weightOffset,bool normalize){
    if(!query.impl_||!key.impl_||!queryWeight.impl_||!keyWeight.impl_||!positions.impl_||!frequencies.impl_||!queryOut.impl_||!keyOut.impl_)return false;
    if(!queryHeads||queryHeads>128||!keyHeads||keyHeads>128||!dim||dim>4096||!rotary||rotary%2||rotary>dim||!(epsilon>0)||!std::isfinite(epsilon)||!std::isfinite(weightOffset))return false;
    auto q=query.impl_->matrix,k=key.impl_->matrix,qw=queryWeight.impl_->matrix,kw=keyWeight.impl_->matrix,p=positions.impl_->matrix,f=frequencies.impl_->matrix,qo=queryOut.impl_->matrix,ko=keyOut.impl_->matrix;
    if(q.rows>160||q.rows!=k.rows||q.columns!=queryHeads*dim||k.columns!=keyHeads*dim||qw.rows!=1||kw.rows!=1||qw.columns!=dim||kw.columns!=dim||p.rows!=q.rows||p.columns!=3||f.rows!=2||f.columns!=rotary/2||qo.rows!=q.rows||ko.rows!=k.rows||qo.columns!=q.columns||ko.columns!=k.columns)return false;
    for(auto m:{q,k,qw,kw,p,f,qo,ko})if(m.device!=impl_->device||m.dataType!=MPSDataTypeFloat32||m.offset)return false;
    if(qo.data==ko.data)throw std::invalid_argument("Paired norm-RoPE outputs alias");
    for(auto input:{q,k,qw,kw,p,f})if(qo.data==input.data||ko.data==input.data)throw std::invalid_argument("Paired norm-RoPE input/output alias");
    if(!impl_->command)throw std::logic_error("Begin a paired norm-RoPE batch first");
    for(auto input:{query.impl_,key.impl_,queryWeight.impl_,keyWeight.impl_,positions.impl_,frequencies.impl_})if(!input->ready&&!(input->pending&&impl_->pendingOutputs.count(input.get())))throw std::logic_error("Uninitialized paired norm-RoPE input");
    for(auto output:{queryOut.impl_,keyOut.impl_})if(output->pending&&!impl_->pendingOutputs.count(output.get()))throw std::logic_error("Foreign paired norm-RoPE output writer");
    struct Rope{uint64_t batch,heads,dim,rotary,iStride,pStride,fStride,oStride;float epsilon,weightOffset;uint64_t normalize;};
    struct Params{Rope query,key;uint64_t queryGroups;};
    Params params{{q.rows,queryHeads,dim,rotary,q.rowBytes/4,p.rowBytes/4,f.rowBytes/4,qo.rowBytes/4,epsilon,weightOffset,uint64_t(normalize)},{k.rows,keyHeads,dim,rotary,k.rowBytes/4,p.rowBytes/4,f.rowBytes/4,ko.rowBytes/4,epsilon,weightOffset,uint64_t(normalize)},q.rows*queryHeads};
    auto pipeline=impl_->kernels->pairedNormRope;if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<32)throw std::runtime_error("Paired norm-RoPE requires original 32-lane SIMD layout");
    auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Paired norm-RoPE encoder failed");[encoder setComputePipelineState:pipeline];
    std::array<MetalMatrix,8> bindings{query,queryWeight,key,keyWeight,positions,frequencies,queryOut,keyOut};for(size_t i=0;i<bindings.size();++i)[encoder setBuffer:bindings[i].impl_->matrix.data offset:0 atIndex:i];
    [encoder setBytes:&params length:sizeof(params) atIndex:8];[encoder dispatchThreadgroups:MTLSizeMake(q.rows*(queryHeads+keyHeads),1,1) threadsPerThreadgroup:MTLSizeMake(32,1,1)];[encoder endEncoding];
    for(auto output:{queryOut.impl_,keyOut.impl_}){output->ready=false;output->pending=true;if(impl_->pendingOutputs.insert(output.get()).second)impl_->outputs.push_back(output);}++impl_->pairedNormRopeDispatches;return true;
}
size_t MetalContext::LastPairedNormRopeDispatches()const{return impl_->pairedNormRopeDispatches;}
'''
def header(source):
 needle='    size_t LastJoinedGlobalMlpBlocks() const;';assert source.count(needle)==1
 return source.replace(needle,needle+'''
    // Private component: identical original Q/K per-head body, one dispatch.
    bool EncodePairedNormRope(const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,const MetalMatrix&,size_t,size_t,size_t,size_t,float,float,bool);
    size_t LastPairedNormRopeDispatches()const;''')
def shader(source):
 start=source.index('kernel void norm_rope(');end=source.index('\nstruct KVParams ',start);original=source[start:end];body=original[original.index('    if(id>='):]
 # Keep every original arithmetic statement, SIMD reduction and precise trig call.
 return '''struct PairedNormRopeParams{RopeParams query,key;ulong queryGroups;};
kernel void paired_norm_rope(device const float* query [[buffer(0)]],device const float* queryWeight [[buffer(1)]],
 device const float* key [[buffer(2)]],device const float* keyWeight [[buffer(3)]],device const float* positions [[buffer(4)]],device const float* frequencies [[buffer(5)]],
 device float* queryOut [[buffer(6)]],device float* keyOut [[buffer(7)]],constant PairedNormRopeParams& pair [[buffer(8)]],uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
 bool isKey=group>=pair.queryGroups;uint id=isKey?group-uint(pair.queryGroups):group;
 RopeParams p;if(isKey)p=pair.key;else p=pair.query;
 device const float* input=isKey?key:query;device const float* weight=isKey?keyWeight:queryWeight;device float* output=isKey?keyOut:queryOut;
'''+body+'\n'
def context(source):
 paired=shader(source)
 def replace(old,new,count=1):
  nonlocal source
  assert source.count(old)==count,(old,source.count(old));source=source.replace(old,new)
 replace('id<MTLComputePipelineState> silu;','id<MTLComputePipelineState> silu,pairedNormRope;')
 replace('    size_t joinedGlobalMlpBlocks=0;','    size_t joinedGlobalMlpBlocks=0,pairedNormRopeDispatches=0;')
 replace('kernel void norm_rope(',paired+'kernel void norm_rope(')
 needle='impl_->kernels->normRope=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"norm_rope"] error:&error];'
 replace(needle,needle+'\n        impl_->kernels->pairedNormRope=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"paired_norm_rope"] error:&error];\n        if(!impl_->kernels->pairedNormRope)throw std::runtime_error("Paired norm-RoPE pipeline unavailable");')
 replace('void MetalContext::EncodeNormRope(',METHODS+'\nvoid MetalContext::EncodeNormRope(')
 replace('impl_->joinedGlobalMlpBlocks=0;','impl_->joinedGlobalMlpBlocks=0;impl_->pairedNormRopeDispatches=0;',2)
 return source
