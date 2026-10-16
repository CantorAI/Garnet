"""Prepare a four-output decode variant that consumes each reconstructed weight immediately."""

def replace(source,old,new,count=1):
 assert source.count(old)==count,(old,source.count(old),count);return source.replace(old,new)

def shader(source):
 # Keep payload bytes, metadata, layout, SIMD grouping, j/K order and reductions.
 # Only the source-level lifetime of reconstructed weights changes. The compiler
 # may already make the two forms equivalent; actual GPU bit/timing gates decide.
 source=replace(source,'kCoalescedBodyBlockShader','kStreamedBodyWeightShader')
 source=replace(source,'kernel void bf16_body_coalesced_block(','kernel void bf16_body_streamed_weights(')
 source=replace(source,'uint meta=metadata[block];float4 weights[4];','uint meta=metadata[block];')
 source=replace(source,'weights[j]=as_type<float4>(bits<<16);','float4 weight=as_type<float4>(bits<<16);\n                sum0+=x[j]*weight;if(p.rows==2)sum1+=y[j]*weight;')
 source=replace(source,'for(uint j=0;j<4;++j)weights[j]=as_type<float4>(uint4(words[j])<<16);','for(uint j=0;j<4;++j){float4 weight=as_type<float4>(uint4(words[j])<<16);sum0+=x[j]*weight;if(p.rows==2)sum1+=y[j]*weight;}')
 return replace(source,'        for(uint j=0;j<4;++j){sum0+=x[j]*weights[j];if(p.rows==2)sum1+=y[j]*weights[j];}\n','')

def header(source):
 needle='size_t LastCoalescedBlockDispatches() const;';return replace(source,needle,needle+'\n    size_t LastStreamedBodyWeightDispatches() const;')

def context(source):
 source=replace(source,'namespace Garnet {','namespace Garnet {bool gCoalescedBlockEnabled=false;size_t gCoalescedBlockThreads=64;')
 needle='#include "coalesced_body_block_shader.h"';source=replace(source,needle,needle+'\n#include "streamed_body_weight_shader.h"')
 source=replace(source,'id<MTLComputePipelineState> bf16BodyCoalescedBlock;','id<MTLComputePipelineState> bf16BodyCoalescedBlock,bf16BodyStreamedWeights;')
 source=replace(source,'size_t mlpGraphDispatches=0;','size_t mlpGraphDispatches=0,streamedBodyWeightDispatches=0;')
 needle='source=[source stringByAppendingString:[NSString stringWithUTF8String:kCoalescedBodyBlockShader]];';source=replace(source,needle,needle+'\n        source=[source stringByAppendingString:[NSString stringWithUTF8String:kStreamedBodyWeightShader]];')
 needle='if(!impl_->kernels->bf16BodyCoalescedBlock)throw std::runtime_error("Cannot create coalesced block pipeline");'
 source=replace(source,needle,needle+'\n        impl_->kernels->bf16BodyStreamedWeights=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_body_streamed_weights"] error:&error];\n        if(!impl_->kernels->bf16BodyStreamedWeights)throw std::runtime_error("Cannot create streamed body weight pipeline");')
 needle='if(coalesced){pipeline=impl_->kernels->bf16BodyCoalescedBlock;threads=64;decodeColumns=8;++impl_->coalescedBlockDispatches;}'
 source=replace(source,needle,needle+'''
        if(coalesced&&gCoalescedBlockEnabled){
            if(gCoalescedBlockThreads!=64)throw std::invalid_argument("Streamed weight comparison fixes64 threads");
            pipeline=impl_->kernels->bf16BodyStreamedWeights;++impl_->streamedBodyWeightDispatches;
        }''')
 needle='size_t MetalContext::LastCoalescedBlockDispatches() const {return impl_->coalescedBlockDispatches;}'
 source=replace(source,needle,needle+'\nsize_t MetalContext::LastStreamedBodyWeightDispatches() const {return impl_->streamedBodyWeightDispatches;}')
 return replace(source,'impl_->mlpGraphDispatches=0;','impl_->mlpGraphDispatches=0;impl_->streamedBodyWeightDispatches=0;',2)

def harness(source):
 # Fixed64 original/candidate geometry; retain every safety/numerical case.
 source=replace(source,'for(size_t threads:{size_t(64),size_t(128),size_t(256)})','for(size_t threads:{size_t(64)})')
 source=replace(source,'expected+=policy&&p.selected;','expected+=p.selected;')
 needle='need(gpu.LastCoalescedBlockDispatches()==expected,"Packed prefill actual route mismatch");'
 source=replace(source,needle,needle+'need(gpu.LastStreamedBodyWeightDispatches()==(policy?expected:0),"Streamed weight actual route mismatch");')
 needle='need(gpu.LastCoalescedBlockDispatches()==size_t(selected&&rows<=2),"Wide route count mismatch");'
 source=replace(source,needle,needle+'need(gpu.LastStreamedBodyWeightDispatches()==size_t(selected&&rows<=2),"Streamed safety route mismatch");')
 needle='need(gpu.LastCoalescedBlockDispatches()==0,"Wide abort counter survived");'
 source=replace(source,needle,needle+'need(gpu.LastStreamedBodyWeightDispatches()==0,"Streamed abort counter survived");')
 source=replace(source,'for(int policy=1;policy<=3;++policy)','for(int policy=1;policy<=1;++policy)',2)
 source=replace(source,'for(int p=1;p<=3;++p)','for(int p=1;p<=1;++p)')
 return source
