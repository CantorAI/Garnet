"""Private storage padding; retain logical shapes and test MPS row alignment only."""
def header(source):
    needle='    size_t LastInterleavedBodyDispatches() const;';assert source.count(needle)==1
    return source.replace(needle,needle+'\n    size_t LastAlignedRowPrefillDispatches() const;\n    bool AlignedRowTailZero(const MetalMatrix&) const;')

def context(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,old[:100]
        source=source.replace(old,new)
    replace('namespace Garnet {','namespace Garnet { int gAlignedRowPrefillPolicy=0;')
    replace('    size_t interleavedBodyDispatches=0;','    size_t interleavedBodyDispatches=0,alignedRowPrefillDispatches=0;')
    old='    id<MTLBuffer> buffer=[impl_->device newBufferWithLength:rows*stride options:MTLResourceStorageModeShared];'
    replace(old,'''    const size_t physicalRows=rows>2&&rows<=160?(rows+127)/128*128:rows;
    if(physicalRows>impl_->device.maxBufferLength/stride)throw std::runtime_error("Aligned storage overflow");
    id<MTLBuffer> buffer=[impl_->device newBufferWithLength:physicalRows*stride options:MTLResourceStorageModeShared];''')
    replace('    if(!buffer)throw std::runtime_error("Metal matrix allocation failed");','''    if(!buffer)throw std::runtime_error("Metal matrix allocation failed");
    if(physicalRows>rows)std::memset(static_cast<char*>(buffer.contents)+rows*stride,0,(physicalRows-rows)*stride);''')
    needle='size_t MetalContext::LastInterleavedBodyDispatches() const {return impl_->interleavedBodyDispatches;}'
    replace(needle,needle+'''
size_t MetalContext::LastAlignedRowPrefillDispatches() const {return impl_->alignedRowPrefillDispatches;}
bool MetalContext::AlignedRowTailZero(const MetalMatrix& input) const {
    if(impl_->command||!input.impl_||!input.impl_->ready)throw std::logic_error("Tail inspection requires a ready idle matrix");
    auto m=input.impl_->matrix;if(m.device!=impl_->device||m.offset!=0)throw std::invalid_argument("Tail inspection device/offset mismatch");
    auto data=static_cast<const float*>(m.data.contents);
    for(size_t i=m.rows*m.rowBytes/4;i<m.data.length/4;++i)if(data[i]!=0.0f)return false;
    return true;
}
''')
    needle='            if(w->globalHalf) {'
    replace(needle,needle+'''
                if(gAlignedRowPrefillPolicy&&a.rows>2&&!w->visionPrefill) {
                    if(gAlignedRowPrefillPolicy<1||gAlignedRowPrefillPolicy>3)throw std::invalid_argument("Aligned row policy invalid");
                    size_t quantum=size_t(16)<<gAlignedRowPrefillPolicy,physical=(a.rows+quantum-1)/quantum*quantum;
                    if(physical>a.rows) {
                        if(a.offset||o.offset||physical>a.data.length/a.rowBytes||physical>o.data.length/o.rowBytes)throw std::runtime_error("Aligned rows exceed owned storage");
                        auto ad=[MPSMatrixDescriptor matrixDescriptorWithRows:physical columns:a.columns rowBytes:a.rowBytes dataType:MPSDataTypeFloat32];
                        auto od=[MPSMatrixDescriptor matrixDescriptorWithRows:physical columns:o.columns rowBytes:o.rowBytes dataType:MPSDataTypeFloat32];
                        a=[[MPSMatrix alloc] initWithBuffer:a.data descriptor:ad];o=[[MPSMatrix alloc] initWithBuffer:o.data descriptor:od];
                        ++impl_->alignedRowPrefillDispatches;
                    }
                }
''')
    assert source.count('impl_->interleavedBodyDispatches=0;')==2
    return source.replace('impl_->interleavedBodyDispatches=0;','impl_->interleavedBodyDispatches=0;impl_->alignedRowPrefillDispatches=0;')
