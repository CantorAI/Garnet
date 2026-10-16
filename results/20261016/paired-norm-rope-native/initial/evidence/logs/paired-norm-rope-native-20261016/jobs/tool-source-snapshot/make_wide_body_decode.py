"""Private eight-output layout; original four-output native library is the control."""
from pathlib import Path
def header(source):
    needle='    bool InterleavedBodyReady() const;';assert source.count(needle)==1
    source=source.replace(needle,needle+'\n    bool WideBodyReady() const;')
    needle='    size_t LastInterleavedBodyDispatches() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'\n    size_t LastWideBodyDispatches() const;')
def context(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old[:100]
        source=source.replace(old,new)
    shader=Path(__file__).with_name('wide_bf16_body_shader.h').read_text()
    shader=shader[shader.index('inline constexpr'):]
    replace('namespace Garnet {','namespace Garnet { bool gWideBodyPacking=false;size_t gWideBodyThreads=128;\n'+shader)
    replace('    id<MTLBuffer> interleavedBody;','    id<MTLBuffer> interleavedBody;bool wideBodyPacking=false;')
    replace('    size_t interleavedBodyDispatches=0;','    size_t interleavedBodyDispatches=0,wideBodyDispatches=0;')
    replace('    id<MTLComputePipelineState> bf16BodyInterleaved;','    id<MTLComputePipelineState> bf16BodyInterleaved,bf16BodyWide;')
    needle='        auto library=[impl_->device newLibraryWithSource:source options:compileOptions error:&error];'
    replace(needle,'        source=[source stringByAppendingString:[NSString stringWithUTF8String:kWideBF16BodyShader]];\n'+needle)
    needle='        if(!impl_->kernels->bf16BodyInterleaved)throw std::runtime_error("Cannot create interleaved BF16 body pipeline");'
    replace(needle,needle+'\n        impl_->kernels->bf16BodyWide=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_body_wide"] error:&error];\n        if(!impl_->kernels->bf16BodyWide)throw std::runtime_error("Cannot create wide body pipeline");')
    start=source.index('        // Retained component policy: large body matrices only.')
    end=source.index('#endif',start)
    block=source[start:end]
    assert block.count('const size_t padded=(rows+3)/4*4;')==1
    block=block.replace('const size_t padded=(rows+3)/4*4;','const size_t width=gWideBodyPacking?8:4,padded=(rows+width-1)/width*width;')
    assert block.count('(c/4)*columns*4+k*4+c%4')==2
    block=block.replace('(c/4)*columns*4+k*4+c%4','(c/width)*columns*width+k*width+c%width')
    block=block.replace('                if(!value->interleavedBody)', '                value->wideBodyPacking=gWideBodyPacking;\n                if(!value->interleavedBody)')
    source=source[:start]+block+source[end:]
    needle='bool MetalBF16Weight::InterleavedBodyReady() const {'
    replace(needle,'bool MetalBF16Weight::WideBodyReady() const {return impl_&&impl_->interleavedBody&&impl_->wideBodyPacking;}\n'+needle)
    needle='size_t MetalContext::LastInterleavedBodyDispatches() const {return impl_->interleavedBodyDispatches;}'
    replace(needle,needle+'\nsize_t MetalContext::LastWideBodyDispatches() const {return impl_->wideBodyDispatches;}')
    needle='        if(interleaved){pipeline=impl_->kernels->bf16BodyInterleaved;threads=128;decodeColumns=16;}'
    replace(needle,needle+'''
        if(interleaved&&w->wideBodyPacking) {
            if(gWideBodyThreads!=64&&gWideBodyThreads!=128&&gWideBodyThreads!=256)throw std::invalid_argument("Wide body group size invalid");
            pipeline=impl_->kernels->bf16BodyWide;threads=gWideBodyThreads;decodeColumns=threads/32*8;
            ++impl_->wideBodyDispatches;
        }
''')
    assert source.count('impl_->interleavedBodyDispatches=0;')==2
    return source.replace('impl_->interleavedBodyDispatches=0;','impl_->interleavedBodyDispatches=0;impl_->wideBodyDispatches=0;')
