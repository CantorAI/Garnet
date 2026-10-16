"""Tune only workgroup geometry in the same original four-output pipeline."""
from make_two_matrix_probe import harness as matrix_harness
from make_coalesced_eight_body import header as extra_header

def context(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
    replace('namespace Garnet {','namespace Garnet {bool gCoalescedBlockEnabled=false;size_t gCoalescedBlockThreads=64;')
    replace('    size_t mlpGraphDispatches=0;','    size_t mlpGraphDispatches=0,tunedBodyDispatches=0;')
    needle='        if(coalesced){pipeline=impl_->kernels->bf16BodyCoalescedBlock;threads=64;decodeColumns=8;++impl_->coalescedBlockDispatches;}'
    replace(needle,needle+'''
        if(coalesced&&gCoalescedBlockEnabled){
            if(gCoalescedBlockThreads!=64&&gCoalescedBlockThreads!=128&&gCoalescedBlockThreads!=256)throw std::invalid_argument("Tuned body threads invalid");
            threads=gCoalescedBlockThreads;decodeColumns=threads/32*4;++impl_->tunedBodyDispatches;
        }
''')
    needle='size_t MetalContext::LastCoalescedBlockDispatches() const {return impl_->coalescedBlockDispatches;}'
    replace(needle,needle+'\nsize_t MetalContext::LastTunedBodyDispatches() const {return impl_->tunedBodyDispatches;}')
    assert source.count('impl_->mlpGraphDispatches=0;')==2
    return source.replace('impl_->mlpGraphDispatches=0;','impl_->mlpGraphDispatches=0;impl_->tunedBodyDispatches=0;')

def header(source):return extra_header(source).replace('LastEightBodyDispatches','LastTunedBodyDispatches')
def harness(source):return matrix_harness(source).replace('LastTwoBodyDispatches','LastTunedBodyDispatches')
def shader(source):return source  # Preserved proof copy only; context includes the original header.
