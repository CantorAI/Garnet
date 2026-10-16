"""Generate an isolated four-output BF16 layout; production sources stay untouched."""
def header(source):
    needle='    bool GlobalHalfReady() const;';assert source.count(needle)==1
    source=source.replace(needle,needle+'\n    bool InterleavedBodyReady() const;')
    needle='    size_t LastGlobalHalfAlphaDispatches() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'\n    size_t LastInterleavedBodyDispatches() const;')

def context(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old[:100]
        source=source.replace(old,new)
    replace('namespace Garnet {','namespace Garnet { bool gInterleavedBody=false;')
    replace('    id<MTLBuffer> buffer;','    id<MTLBuffer> buffer,interleavedBody;')
    replace('bool MetalBF16Weight::GlobalHalfReady() const {return ExactHalfReady()&&impl_->globalHalf;}',
            'bool MetalBF16Weight::GlobalHalfReady() const {return ExactHalfReady()&&impl_->globalHalf;}\nbool MetalBF16Weight::InterleavedBodyReady() const {return impl_&&impl_->interleavedBody;}')
    replace('bf16Decode,bf16DecodeSmall,','bf16Decode,bf16BodyInterleaved,bf16DecodeSmall,')
    replace('    size_t globalHalfAlphaDispatches=0;','    size_t globalHalfAlphaDispatches=0,interleavedBodyDispatches=0;')
    start=source.index('kernel void bf16_decode(');end=source.index('kernel void bf16_decode_small(',start)
    kernel=source[start:end].replace('kernel void bf16_decode(','kernel void bf16_body_interleaved(')
    a=kernel.index('            ushort4 b0=');b=kernel.index('                sum0+=',a)
    kernel=kernel[:a]+'''            ulong packed=(column/4)*p.inner*4+k*4;
            uint4 pair0=*(device const uint4*)(weight+packed),pair1=*(device const uint4*)(weight+packed+8);
            ushort4 words[4]={as_type<ushort4>(pair0.xy),as_type<ushort4>(pair0.zw),as_type<ushort4>(pair1.xy),as_type<ushort4>(pair1.zw)};
            for(uint j=0;j<4;++j) {
                float4 w=as_type<float4>(uint4(words[j])<<16);
'''+kernel[b:]
    # The unchanged odd-K branch remains unreachable for this private layout.
    source=source[:start]+kernel+source[start:]
    needle='        impl_->kernels->bf16Decode=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_decode"] error:&error];'
    replace(needle,needle+'\n        impl_->kernels->bf16BodyInterleaved=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_body_interleaved"] error:&error];\n        if(!impl_->kernels->bf16BodyInterleaved)throw std::runtime_error("Interleaved body pipeline unavailable");')
    needle='        if(!value->buffer)throw std::runtime_error("BF16 weight allocation failed");'
    replace(needle,needle+'''
        if(!visionPrefill&&!floatPrefill&&rows>256&&rows<=8192&&columns%4==0) {
            const size_t padded=(rows+3)/4*4;std::vector<uint16_t> packed(padded*columns,0);
            auto raw=static_cast<const char*>(data);
            for(size_t c=0;c<rows;++c)for(size_t k=0;k<columns;++k)
                std::memcpy(&packed[(c/4)*columns*4+k*4+c%4],raw+2*(c*columns+k),2);
            for(size_t c=0;c<padded;++c)for(size_t k=0;k<columns;++k) {
                uint16_t expected=0;if(c<rows)std::memcpy(&expected,raw+2*(c*columns+k),2);
                if(packed[(c/4)*columns*4+k*4+c%4]!=expected)throw std::runtime_error("BF16 CPU layout roundtrip failed");
            }
            value->interleavedBody=[impl_->device newBufferWithBytes:packed.data() length:packed.size()*2 options:MTLResourceStorageModeShared];
            if(!value->interleavedBody)throw std::runtime_error("Interleaved body allocation failed");
        }
''')
    needle='size_t MetalContext::LastGlobalHalfAlphaDispatches() const {return impl_->globalHalfAlphaDispatches;}'
    replace(needle,needle+'\nsize_t MetalContext::LastInterleavedBodyDispatches() const {return impl_->interleavedBodyDispatches;}')
    needle='        if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<threads)'
    replace(needle,'''        const bool interleaved=gInterleavedBody&&a.rows<=2&&w->interleavedBody;
        if(interleaved){pipeline=impl_->kernels->bf16BodyInterleaved;threads=128;decodeColumns=16;++impl_->interleavedBodyDispatches;}
'''+needle)
    needle='        [encoder setBuffer:w->buffer offset:0 atIndex:1];[encoder setBuffer:o.data offset:0 atIndex:2];'
    replace(needle,'        [encoder setBuffer:interleaved?w->interleavedBody:w->buffer offset:0 atIndex:1];[encoder setBuffer:o.data offset:0 atIndex:2];')
    assert source.count('impl_->globalHalfAlphaDispatches=0;')==2
    source=source.replace('impl_->globalHalfAlphaDispatches=0;','impl_->globalHalfAlphaDispatches=0;impl_->interleavedBodyDispatches=0;')
    return source

def harness(source):
    source=source.replace('gDirectMPS','gInterleavedBody').replace('p.weight.GlobalHalfReady()','p.weight.InterleavedBodyReady()').replace('LastGlobalHalfAlphaDispatches()','LastInterleavedBodyDispatches()')
    needle='    for(size_t batch:{size_t(1),size_t(2)}){';assert source.count(needle)==1
    source=source.replace(needle,needle+'for(int pattern=0;pattern<4;++pattern){')
    needle='x[r*p.k+j]=std::sin(float((j*17+r*53)%1301)*.013f)*.2f;';assert source.count(needle)==1
    source=source.replace(needle,'''{
                float value=std::sin(float((j*17+r*53)%1301)*.013f)*.2f;
                if(pattern==1)value*=float(j%2?-16:16);
                if(pattern==2)value=j==((r*53+17)%p.k)?1.0f:0.0f;
                if(pattern==3)value=std::ldexp(value,int(j%9)-4);
                x[r*p.k+j]=value;
            }''')
    source=source.replace('bool finite=true;','bool finite=true,bitsExact=true;')
    needle='auto actual=gpu.Download(p.output);values+=actual.size();';assert source.count(needle)==1
    source=source.replace(needle,needle+'bitsExact=bitsExact&&!std::memcmp(actual.data(),p.reference.data(),actual.size()*sizeof(float));')
    needle='admitted[policy]=finite&&worst<=1&&comparison<=1;';assert source.count(needle)==1
    source=source.replace(needle,needle+'need(admitted[policy]&&bitsExact,"Interleaved body numerical or original-bit equality gate failed");')
    source=source.replace('@"forced_mps_projections":@(forced)','@"interleaved_projections":@(forced),@"bitwise_equal":@(bitsExact),@"pattern":@(pattern)')
    needle='        auto warm=std::chrono::steady_clock::now();';assert source.count(needle)==1
    source=source.replace(needle,'        if(pattern!=0)continue;\n'+needle)
    needle='    }return 0;';assert source.count(needle)==1
    source=source.replace(needle,'    }}return 0;').replace('streamed_direct_mps_failed','interleaved_bf16_body_failed')
    needle='int main(int argc,char** argv)';assert source.count(needle)==1
    source=source.replace(needle,'''static void controls(){
    Garnet::MetalContext gpu,other;size_t checks=0;
    for(auto shape:std::vector<std::pair<size_t,size_t>>{{257,128},{256,128},{8193,128},{257,129}}){
        size_t n=shape.first,k=shape.second;std::vector<uint16_t> bits(n*k,0x3e80);
        auto weight=gpu.UploadBF16(bits.data(),bits.size()*2,n,k);
        bool admitted=n>256&&n<=8192&&k%4==0;need(weight.InterleavedBodyReady()==admitted,"Body shape admission mismatch");
        auto input=gpu.Upload(std::vector<float>(2*k,.125f),2,k),out=gpu.Allocate(2,n);
        Garnet::gInterleavedBody=false;gpu.Begin();gpu.EncodeBF16MatVec(input,weight,out);gpu.SubmitAndWait();auto raw=gpu.Download(out);
        Garnet::gInterleavedBody=true;gpu.Begin();gpu.EncodeBF16MatVec(input,weight,out);bool rejected=false;
        try{gpu.Download(out);}catch(const std::logic_error&){rejected=true;}need(rejected,"Pending body output read");
        other.Begin();rejected=false;try{other.EncodeBF16MatVec(input,weight,out);}catch(const std::logic_error&){rejected=true;}
        need(rejected,"Foreign command body output accepted");other.Abort();gpu.SubmitAndWait();auto actual=gpu.Download(out);
        need(actual.size()==raw.size()&&!std::memcmp(actual.data(),raw.data(),actual.size()*4),"Body shape/stride bits changed");
        need(gpu.LastInterleavedBodyDispatches()==size_t(admitted),"Body fallback count mismatch");
        gpu.Begin();gpu.EncodeBF16MatVec(input,weight,out);gpu.Abort();need(gpu.LastInterleavedBodyDispatches()==0,"Aborted body count survived");
        rejected=false;try{gpu.Download(out);}catch(const std::logic_error&){rejected=true;}need(rejected,"Aborted body readable");
        auto uninitialized=gpu.Allocate(2,k);gpu.Begin();rejected=false;
        try{gpu.EncodeBF16MatVec(uninitialized,weight,out);}catch(const std::logic_error&){rejected=true;}
        need(rejected&&gpu.LastInterleavedBodyDispatches()==0,"Uninitialized body input accepted");gpu.Abort();
        gpu.Begin();gpu.EncodeBF16MatVec(input,weight,out);gpu.SubmitAndWait();need(gpu.Download(out)==raw,"Body abort recovery changed values");++checks;
    }
    emit(@{@"kind":@"controls",@"profiles":@(checks),@"passed":@YES});
}
'''+needle)
    needle='Garnet::MetalContext gpu;std::vector<Projection> projections;';assert source.count(needle)==1
    return source.replace(needle,'controls();'+needle)
