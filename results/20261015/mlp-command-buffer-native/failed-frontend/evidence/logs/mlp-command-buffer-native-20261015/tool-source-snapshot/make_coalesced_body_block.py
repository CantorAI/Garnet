"""Exact 12-bit fixed-pitch four-output blocks, native raw fallback for wide exponents."""
from pathlib import Path
def header(source):
    needle='    bool InterleavedBodyReady() const;';assert source.count(needle)==1
    source=source.replace(needle,needle+'\n    bool CoalescedBlockReady() const;\n    size_t CoalescedBlockCount() const;\n    size_t CoalescedCompactCount() const;')
    needle='    size_t LastInterleavedBodyDispatches() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'\n    size_t LastCoalescedBlockDispatches() const;')
def context(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old[:100];source=source.replace(old,new)
    shader=Path(__file__).with_name('coalesced_body_block_shader.h').read_text();shader=shader[shader.index('inline constexpr'):]
    replace('namespace Garnet {','namespace Garnet {bool gCoalescedBlockEnabled=false;size_t gCoalescedBlockThreads=128;\n'+shader)
    replace('    id<MTLBuffer> interleavedBody;','    id<MTLBuffer> interleavedBody,coalescedPayload,coalescedMetadata;size_t coalescedBlocks=0,coalescedCompact=0;')
    replace('    size_t interleavedBodyDispatches=0;','    size_t interleavedBodyDispatches=0,coalescedBlockDispatches=0;')
    replace('    id<MTLComputePipelineState> bf16BodyInterleaved;','    id<MTLComputePipelineState> bf16BodyInterleaved,bf16BodyCoalescedBlock;')
    needle='        auto library=[impl_->device newLibraryWithSource:source options:compileOptions error:&error];'
    replace(needle,'        source=[source stringByAppendingString:[NSString stringWithUTF8String:kCoalescedBodyBlockShader]];\n'+needle)
    needle='        if(!impl_->kernels->bf16BodyInterleaved)throw std::runtime_error("Cannot create interleaved BF16 body pipeline");'
    replace(needle,needle+'\n        impl_->kernels->bf16BodyCoalescedBlock=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_body_coalesced_block"] error:&error];\n        if(!impl_->kernels->bf16BodyCoalescedBlock)throw std::runtime_error("Cannot create coalesced block pipeline");')
    needle='                if(!value->interleavedBody)throw std::runtime_error("Interleaved BF16 body allocation failed");'
    replace(needle,needle+'''
                if(columns%128==0) {
                    size_t blocks=padded/4*(columns/128);
                    if(blocks>impl_->device.maxBufferLength/768||blocks>impl_->device.maxBufferLength/4)throw std::runtime_error("Coalesced block storage overflow");
                    std::vector<uint8_t> payload(blocks*768,0);std::vector<uint32_t> metadata(blocks,0);
                    size_t compact=0;
                    for(size_t group=0;group<padded/4;++group)for(size_t block=0;block<columns/128;++block) {
                        size_t index=group*(columns/128)+block;uint16_t low=255,high=0;
                        for(size_t k=0;k<128;++k)for(size_t j=0;j<4;++j){uint16_t bits=packed[group*columns*4+(block*128+k)*4+j],e=(bits>>7)&255;low=std::min(low,e);high=std::max(high,e);}
                        if(high-low>15)continue;
                        metadata[index]=uint32_t(low)|256u;++compact;
                        for(size_t k=0;k<128;++k)for(size_t j=0;j<4;++j){size_t position=k*4+j,at=index*768;
                            uint16_t bits=packed[group*columns*4+(block*128+k)*4+j];payload[at+position]=uint8_t((bits&127)|((bits>>8)&128));
                            uint8_t tag=uint8_t(((bits>>7)&255)-low);payload[at+512+position/2]|=uint8_t(tag<<((position%2)*4));}
                        for(size_t k=0;k<128;++k)for(size_t j=0;j<4;++j){size_t position=k*4+j,at=index*768;uint16_t sm=payload[at+position],tag=(payload[at+512+position/2]>>((position%2)*4))&15;
                            uint16_t bits=uint16_t((sm&127)|((sm&128)<<8)|((tag+low)<<7));
                            if(bits!=packed[group*columns*4+(block*128+k)*4+j])throw std::runtime_error("Coalesced block original-bit roundtrip failed");}
                    }
                    value->coalescedBlocks=blocks;value->coalescedCompact=compact;
                    if(compact){value->coalescedPayload=[impl_->device newBufferWithBytes:payload.data() length:payload.size() options:MTLResourceStorageModeShared];value->coalescedMetadata=[impl_->device newBufferWithBytes:metadata.data() length:metadata.size()*4 options:MTLResourceStorageModeShared];if(!value->coalescedPayload||!value->coalescedMetadata)throw std::runtime_error("Coalesced block allocation failed");}
                }
''')
    needle='bool MetalBF16Weight::InterleavedBodyReady() const {'
    replace(needle,'''bool MetalBF16Weight::CoalescedBlockReady() const {return impl_&&impl_->coalescedPayload&&impl_->coalescedMetadata;}
size_t MetalBF16Weight::CoalescedBlockCount() const {return impl_?impl_->coalescedBlocks:0;}
size_t MetalBF16Weight::CoalescedCompactCount() const {return impl_?impl_->coalescedCompact:0;}
'''+needle)
    needle='size_t MetalContext::LastInterleavedBodyDispatches() const {return impl_->interleavedBodyDispatches;}'
    replace(needle,needle+'\nsize_t MetalContext::LastCoalescedBlockDispatches() const {return impl_->coalescedBlockDispatches;}')
    needle='        if(interleaved){pipeline=impl_->kernels->bf16BodyInterleaved;threads=128;decodeColumns=16;}'
    replace(needle,needle+'''
        const bool coalesced=interleaved&&gCoalescedBlockEnabled&&w->coalescedPayload;
        if(coalesced){if(gCoalescedBlockThreads!=64&&gCoalescedBlockThreads!=128&&gCoalescedBlockThreads!=256)throw std::invalid_argument("Coalesced block threads invalid");pipeline=impl_->kernels->bf16BodyCoalescedBlock;threads=gCoalescedBlockThreads;decodeColumns=threads/32*4;++impl_->coalescedBlockDispatches;}
''')
    needle='        if(interleaved)++impl_->interleavedBodyDispatches;'
    replace(needle,needle+'\n        if(coalesced){[encoder setBuffer:w->coalescedPayload offset:0 atIndex:4];[encoder setBuffer:w->coalescedMetadata offset:0 atIndex:5];}')
    assert source.count('impl_->interleavedBodyDispatches=0;')==2
    return source.replace('impl_->interleavedBodyDispatches=0;','impl_->interleavedBodyDispatches=0;impl_->coalescedBlockDispatches=0;')
