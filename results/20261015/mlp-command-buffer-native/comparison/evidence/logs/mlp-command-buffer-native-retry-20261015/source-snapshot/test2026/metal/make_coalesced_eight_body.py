"""Private current compact four-output control versus eight-output scheduling."""
from pathlib import Path

def header(source):
    needle='    size_t LastCoalescedBlockDispatches() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'\n    size_t LastEightBodyDispatches() const;')

def context(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    replace('namespace Garnet {','namespace Garnet {bool gCoalescedBlockEnabled=false;size_t gCoalescedBlockThreads=64;')
    replace('#include "coalesced_body_block_shader.h"','#include "coalesced_body_block_shader.h"\n#include "coalesced_eight_body_shader.h"')
    replace('    id<MTLComputePipelineState> bf16BodyCoalescedBlock;',
            '    id<MTLComputePipelineState> bf16BodyCoalescedBlock,bf16BodyCoalescedEight;')
    replace('    size_t mlpGraphDispatches=0;','    size_t mlpGraphDispatches=0,eightBodyDispatches=0;')
    needle='        source=[source stringByAppendingString:[NSString stringWithUTF8String:kCoalescedBodyBlockShader]];'
    replace(needle,needle+'\n        source=[source stringByAppendingString:[NSString stringWithUTF8String:kCoalescedEightBodyShader]];')
    needle='        if(!impl_->kernels->bf16BodyCoalescedBlock)throw std::runtime_error("Cannot create coalesced block pipeline");'
    replace(needle,needle+'\n        impl_->kernels->bf16BodyCoalescedEight=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_body_coalesced_eight"] error:&error];\n        if(!impl_->kernels->bf16BodyCoalescedEight)throw std::runtime_error("Cannot create compact eight-output pipeline");')
    needle='        if(coalesced){pipeline=impl_->kernels->bf16BodyCoalescedBlock;threads=64;decodeColumns=8;++impl_->coalescedBlockDispatches;}'
    replace(needle,needle+'''
        if(coalesced&&gCoalescedBlockEnabled){
            if(gCoalescedBlockThreads!=64&&gCoalescedBlockThreads!=128&&gCoalescedBlockThreads!=256)throw std::invalid_argument("Eight body threads invalid");
            pipeline=impl_->kernels->bf16BodyCoalescedEight;threads=gCoalescedBlockThreads;decodeColumns=threads/32*8;++impl_->eightBodyDispatches;
        }
''')
    needle='size_t MetalContext::LastCoalescedBlockDispatches() const {return impl_->coalescedBlockDispatches;}'
    replace(needle,needle+'\nsize_t MetalContext::LastEightBodyDispatches() const {return impl_->eightBodyDispatches;}')
    assert source.count('impl_->mlpGraphDispatches=0;')==2
    return source.replace('impl_->mlpGraphDispatches=0;','impl_->mlpGraphDispatches=0;impl_->eightBodyDispatches=0;')

def harness(source):
    needle='need(gpu.LastCoalescedBlockDispatches()==expected,"Packed prefill actual route mismatch");'
    assert source.count(needle)==1
    source=source.replace(needle,needle+'need(gpu.LastEightBodyDispatches()==(policy?expected:0),"Eight body actual route mismatch");')
    source=source.replace('expected+=policy&&p.selected;','expected+=p.selected;')
    source=source.replace('"Wide route count mismatch");','"Wide route count mismatch");need(gpu.LastEightBodyDispatches()==size_t(selected&&rows<=2),"Eight safety route mismatch");')
    source=source.replace('"Wide abort counter survived");','"Wide abort counter survived");need(gpu.LastEightBodyDispatches()==0,"Eight abort counter survived");')
    assert 'Eight safety route mismatch' in source and 'Eight abort counter survived' in source
    return source
