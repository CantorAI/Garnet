"""Private four-bit rejection index; original BF16/FP32 seed and refinement retained."""
def replace(source,old,new,count=1):
 assert source.count(old)==count,(old[:100],source.count(old),count)
 return source.replace(old,new)

def shader(source):
 source=replace(source,'kScreenedHeadShader','kNibbleScreenedHeadShader')
 source=replace(source,'prepare_head_bounds','nibble_prepare_head_bounds')
 source=replace(source,'kernel void screened_head(','kernel void nibble_screened_head(')
 old='''        uint4 packed=*(device const uint4*)(approximate+column*p.inner+k*4);
        for(uint j=0;j<4;++j){float4 w=float4(as_type<char4>(packed[j]));sum0=fma(x[j],w,sum0);if(p.rows==2)sum1=fma(y[j],w,sum1);}'''
 new='''        ushort4 packed=as_type<ushort4>(*(device const uint2*)(approximate+(column*p.inner+k*4)/2));
        for(uint j=0;j<4;++j){uint4 tags=uint4(packed[j],packed[j]>>4,packed[j]>>8,packed[j]>>12)&15u;
            int4 signedValues=int4(tags)-select(int4(0),int4(16),tags>=8u);float4 w=float4(signedValues);
            sum0=fma(x[j],w,sum0);if(p.rows==2)sum1=fma(y[j],w,sum1);}'''
 return replace(source,old,new)

def index(source):
 source=replace(source,'int8_t* encoded,float* metadata) {','int8_t* encoded,float* metadata,int maximum=127) {')
 source=replace(source,'    double largest=0;','    if(maximum!=7&&maximum!=127)return false;\n    double largest=0;')
 source=replace(source,'largest/127.','largest/double(maximum)')
 return replace(source,'if(q<-127||q>127||','if(q<-maximum||q>maximum||')

def native_impl(source):
 needle='        w->screenAttempted=true;';source=replace(source,needle,needle+'const bool nibble=gNibbleScreenEnabled;')
 source=replace(source,'encoded+row*w->columns,info+row*4)','encoded+row*w->columns,info+row*4,nibble?7:127)')
 start=source.index('        const size_t paddedRows=(w->rows+3)/4*4,packedBytes=')
 end=source.index('        index=packed;',start)
 source=source[:start]+'''        const size_t paddedRows=(w->rows+3)/4*4,packedBytes=Elements(paddedRows,w->columns)/(nibble?2:1);
        if(packedBytes>impl_->device.maxBufferLength)return false;
        auto packed=[impl_->device newBufferWithLength:packedBytes options:MTLResourceStorageModeShared];if(!packed)return false;
        auto destination=static_cast<uint8_t*>(packed.contents);std::memset(destination,0,packedBytes);
        for(size_t group=0;group<paddedRows/4;++group)for(size_t k=0;k<w->columns;++k)for(size_t c=0;c<4;++c){
            const size_t row=group*4+c,at=(group*w->columns+k)*4+c;const int8_t q=row<w->rows?encoded[row*w->columns+k]:0;
            if(nibble)destination[at/2]|=(uint8_t(q)&15u)<<((at%2)*4);else destination[at]=uint8_t(q);
        }
        for(size_t row=0;row<paddedRows;++row)for(size_t k=0;k<w->columns;++k){
            size_t at=((row/4)*w->columns+k)*4+row%4;int value;
            if(nibble){value=(destination[at/2]>>((at%2)*4))&15u;if(value>=8)value-=16;}else value=int8_t(destination[at]);
            if(value!=(row<w->rows?int(encoded[row*w->columns+k]):0))return false;
        }
'''+source[end:]
 source=replace(source,'w->screenIndex=index;w->screenMetadata=metadata;','w->screenIndex=index;w->screenMetadata=metadata;w->nibbleScreen=nibble;')
 source=replace(source,'projection=impl_->kernels->screenProjection,','projection=w->nibbleScreen?impl_->kernels->nibbleScreenProjection:impl_->kernels->screenProjection,')
 needle='if(kScreenedHeadInterleaved)++impl_->interleavedScreenDispatches;return true;'
 source=replace(source,needle,'if(w->nibbleScreen)++impl_->nibbleScreenDispatches;'+needle)
 return source

def header(source):
 source=replace(source,'    bool CoalescedBlockReady() const;','    bool CoalescedBlockReady() const;\n    bool NibbleScreenReady() const;\n    size_t ScreenIndexBytes() const;')
 return replace(source,'    size_t LastScreenedHeadDispatches() const;','    size_t LastScreenedHeadDispatches() const;\n    size_t LastNibbleScreenDispatches() const;')

def context(source):
 source=replace(source,'namespace Garnet {','namespace Garnet {bool gNibbleScreenEnabled=false;')
 source=replace(source,'#include "screened_head_shader.h"','#include "screened_head_shader.h"\n#include "nibble_screened_head_shader.h"')
 source=replace(source,'    bool screenAttempted=false;','    bool screenAttempted=false,nibbleScreen=false;')
 source=replace(source,'screenPrepare,screenProjection;','screenPrepare,screenProjection,nibbleScreenProjection;')
 source=replace(source,'    size_t screenedHeadDispatches=0;','    size_t screenedHeadDispatches=0,nibbleScreenDispatches=0;')
 needle='            impl_->kernels->screenProjection=[impl_->device newComputePipelineStateWithFunction:[screenLibrary newFunctionWithName:@"screened_head"] error:&error];'
 source=replace(source,needle,needle+'''
            NSString* nibbleSource=[NSString stringWithUTF8String:kNibbleScreenedHeadShader];
            auto nibbleLibrary=[impl_->device newLibraryWithSource:nibbleSource options:screenOptions error:&error];
            if(!nibbleLibrary)throw std::runtime_error(error.localizedDescription.UTF8String);
            impl_->kernels->nibbleScreenProjection=[impl_->device newComputePipelineStateWithFunction:[nibbleLibrary newFunctionWithName:@"nibble_screened_head"] error:&error];
            if(!impl_->kernels->nibbleScreenProjection)throw std::runtime_error("Nibble screening pipeline unavailable");
''')
 source=replace(source,'impl_->screenedHeadDispatches=0;','impl_->screenedHeadDispatches=0;impl_->nibbleScreenDispatches=0;',2)
 needle='double MetalContext::LastGPUMs()'
 methods='''bool MetalBF16Weight::NibbleScreenReady() const {return impl_&&impl_->screenIndex&&impl_->nibbleScreen;}
size_t MetalBF16Weight::ScreenIndexBytes() const {return impl_&&impl_->screenIndex?impl_->screenIndex.length:0;}
size_t MetalContext::LastNibbleScreenDispatches() const {return impl_->nibbleScreenDispatches;}
'''
 return replace(source,needle,methods+needle)
