"""Private FP32 prefill tiles reading the validated four-output BF16 layout."""
from make_prefill_tile_shader import kernel
TILES=((16,64,64),(32,64,32),(16,32,128))
def header(source):
    needle='    size_t LastInterleavedBodyDispatches() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'\n    size_t LastPackedBodyPrefillDispatches() const;')
def context(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old[:100];source=source.replace(old,new)
    shaders=[]
    for index,tile in enumerate(TILES):
        code=kernel(tile);old='weight[column*p.inner+k]';assert code.count(old)==1
        code=code.replace(old,'weight[(column/4)*p.inner*4+k*4+column%4]')
        code=code.replace('kernel void gemm_'+'_'.join(map(str,tile))+'(','kernel void packed_body_prefill_'+str(index)+'(')
        shaders.append(code)
    replace('namespace Garnet {','namespace Garnet { int gPackedBodyPrefillPolicy=0;\nstatic constexpr const char* kPackedBodyPrefillShader=R"PREFILL('+''.join(shaders)+')PREFILL";')
    replace('    size_t interleavedBodyDispatches=0;','    size_t interleavedBodyDispatches=0,packedBodyPrefillDispatches=0;')
    replace('    id<MTLComputePipelineState> bf16BodyInterleaved;','    id<MTLComputePipelineState> bf16BodyInterleaved,packedBodyPrefill0,packedBodyPrefill1,packedBodyPrefill2;')
    needle='        auto library=[impl_->device newLibraryWithSource:source options:compileOptions error:&error];'
    replace(needle,'        source=[source stringByAppendingString:[NSString stringWithUTF8String:kPackedBodyPrefillShader]];\n'+needle)
    needle='        if(!impl_->kernels->bf16BodyInterleaved)throw std::runtime_error("Cannot create interleaved BF16 body pipeline");'
    extra=''
    for index in range(3):
        extra+='\n        impl_->kernels->packedBodyPrefill'+str(index)+'=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"packed_body_prefill_'+str(index)+'"] error:&error];'
        extra+='\n        if(!impl_->kernels->packedBodyPrefill'+str(index)+')throw std::runtime_error("Packed prefill pipeline unavailable");'
    replace(needle,needle+extra)
    needle='size_t MetalContext::LastInterleavedBodyDispatches() const {return impl_->interleavedBodyDispatches;}'
    replace(needle,needle+'\nsize_t MetalContext::LastPackedBodyPrefillDispatches() const {return impl_->packedBodyPrefillDispatches;}')
    needle='        if(!w->visionPrefill&&a.rows==160)++impl_->wideBatchPrefillDispatches;'
    replace(needle,needle+'''
        if(gPackedBodyPrefillPolicy&&a.rows>2&&w->interleavedBody) {
            if(gPackedBodyPrefillPolicy<1||gPackedBodyPrefillPolicy>3)throw std::invalid_argument("Packed prefill policy invalid");
            const size_t tr=gPackedBodyPrefillPolicy==2?32:16,tc=gPackedBodyPrefillPolicy==3?32:64,threads=tc/8*32;
            auto pipeline=gPackedBodyPrefillPolicy==1?impl_->kernels->packedBodyPrefill0:gPackedBodyPrefillPolicy==2?impl_->kernels->packedBodyPrefill1:impl_->kernels->packedBodyPrefill2;
            if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<threads||pipeline.staticThreadgroupMemoryLength>impl_->device.maxThreadgroupMemoryLength)throw std::runtime_error("Packed prefill geometry unsupported");
            struct Params{uint64_t rows,inner,columns,aStride,oStride;};Params p{a.rows,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4};
            auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Packed prefill encoder unavailable");
            [encoder setComputePipelineState:pipeline];[encoder setBuffer:a.data offset:0 atIndex:0];[encoder setBuffer:w->interleavedBody offset:0 atIndex:1];[encoder setBuffer:o.data offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((o.columns+tc-1)/tc,(a.rows+tr-1)/tr,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];[encoder endEncoding];
            output.impl_->ready=false;output.impl_->pending=true;if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
            ++impl_->packedBodyPrefillDispatches;return;
        }
''')
    assert source.count('impl_->interleavedBodyDispatches=0;')==2
    return source.replace('impl_->interleavedBodyDispatches=0;','impl_->interleavedBodyDispatches=0;impl_->packedBodyPrefillDispatches=0;')
