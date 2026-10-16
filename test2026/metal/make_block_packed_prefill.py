"""Private K32/N64 BF16 tiles, shared scratch reuse, FP32 SIMD matrices."""
TILES=((32,64,32),(64,64,32))

def replace(source,old,new,count=1):
 assert source.count(old)==count,(old[:100],source.count(old),count)
 return source.replace(old,new)

def kernel(rows):
 assert rows in (32,64)
 lines=[f'kernel void block_packed_prefill_{rows}(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],',
 'device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {',
 f'threadgroup float scratch[{rows*64}],right[2048];',
 f'ulong rowBase=ulong(group.y)*{rows},colBase=ulong(group.x)*64,kTiles=(p.inner+31)/32;']
 for band in range(rows//8):lines.append(f'simdgroup_float8x8 acc{band}=make_filled_simdgroup_matrix<float,8>(0.0f);')
 lines += ['for(ulong base=0;base<p.inner;base+=32){',
 f'for(uint i=tid;i<{rows*32};i+=256){{ulong row=rowBase+i/32,k=base+i%32;scratch[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}}',
 'for(uint i=tid;i<2048;i+=256)right[i]=as_type<float>(uint(weight[(ulong(group.x)*kTiles+base/32)*2048+i])<<16);',
 'threadgroup_barrier(mem_flags::mem_threadgroup);',
 'for(uint offset=0;offset<32;offset+=8){simdgroup_float8x8 r;simdgroup_load(r,right+offset*64+simd*8,64);']
 for band in range(rows//8):
  lines.append(f'if(rowBase+{band*8}<p.rows){{simdgroup_float8x8 l;simdgroup_load(l,scratch+{band*8*32}+offset,32);simdgroup_multiply_accumulate(acc{band},l,r,acc{band});}}')
 lines += ['}','threadgroup_barrier(mem_flags::mem_threadgroup);','}']
 for band in range(rows//8):lines.append(f'if(rowBase+{band*8}<p.rows)simdgroup_store(acc{band},scratch+{band*8*64}+simd*8,64);')
 lines += ['threadgroup_barrier(mem_flags::mem_threadgroup);',f'for(uint i=tid;i<{rows*64};i+=256){{ulong row=rowBase+i/64,column=colBase+i%64;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=scratch[i];}}','}']
 return '\n'.join(lines)+'\n'

def shader():return 'static constexpr const char* kBlockPackedPrefillShader=R"BLOCK('+''.join(kernel(t[0]) for t in TILES)+')BLOCK";\n'

def header(source):
 source=replace(source,'    bool InterleavedBodyReady() const;','    bool InterleavedBodyReady() const;\n    size_t BlockPackedPrefillBytes() const;')
 return replace(source,'    size_t LastInterleavedBodyDispatches() const;','    size_t LastInterleavedBodyDispatches() const;\n    size_t LastBlockPackedPrefillDispatches() const;')

def context(source):
 source=replace(source,'#include "metal_context.h"','#include "metal_context.h"\n#include "block_packed_prefill_shader.h"')
 source=replace(source,'namespace Garnet {','namespace Garnet {int gBlockPackedPrefillPolicy=0;')
 source=replace(source,'    id<MTLBuffer> interleavedBody;','    id<MTLBuffer> interleavedBody,blockPackedPrefill;')
 needle='    id<MTLComputePipelineState> bf16BodyInterleaved;'
 source=replace(source,needle,needle+'\n    id<MTLComputePipelineState> blockPackedPrefill32,blockPackedPrefill64;')
 source=replace(source,'    size_t mlpGraphDispatches=0;','    size_t mlpGraphDispatches=0,blockPackedPrefillDispatches=0;')
 needle='        auto library=[impl_->device newLibraryWithSource:source options:compileOptions error:&error];'
 source=replace(source,needle,'        source=[source stringByAppendingString:[NSString stringWithUTF8String:kBlockPackedPrefillShader]];\n'+needle)
 needle='        if(!impl_->kernels->bf16BodyInterleaved)throw std::runtime_error("Cannot create interleaved BF16 body pipeline");'
 extra=''
 for rows in (32,64):extra+=f'\n        impl_->kernels->blockPackedPrefill{rows}=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"block_packed_prefill_{rows}"] error:&error];\n        if(!impl_->kernels->blockPackedPrefill{rows})throw std::runtime_error("Block packed pipeline unavailable");'
 source=replace(source,needle,needle+extra)
 needle='                if(!value->interleavedBody)throw std::runtime_error("Interleaved BF16 body allocation failed");'
 source=replace(source,needle,needle+'''
                // Additional private copy: fixed N64/K32 blocks, K-major within a block.
                const size_t nTiles=(rows+63)/64,kTiles=(columns+31)/32;
                if(nTiles>impl_->device.maxBufferLength/4096/kTiles)throw std::runtime_error("Block packed storage overflow");
                std::vector<uint16_t> tiled(nTiles*kTiles*2048,0);
                for(size_t n=0;n<rows;++n)for(size_t k=0;k<columns;++k){uint16_t bits;std::memcpy(&bits,raw+2*(n*columns+k),2);tiled[((n/64)*kTiles+k/32)*2048+(k%32)*64+n%64]=bits;}
                for(size_t n=0;n<nTiles*64;++n)for(size_t k=0;k<kTiles*32;++k){uint16_t expected=0;if(n<rows&&k<columns)std::memcpy(&expected,raw+2*(n*columns+k),2);if(tiled[((n/64)*kTiles+k/32)*2048+(k%32)*64+n%64]!=expected)throw std::runtime_error("Block packed bit roundtrip failed");}
                value->blockPackedPrefill=[impl_->device newBufferWithBytes:tiled.data() length:tiled.size()*2 options:MTLResourceStorageModeShared];
                if(!value->blockPackedPrefill)throw std::runtime_error("Block packed allocation failed");
''')
 needle='size_t MetalContext::LastInterleavedBodyDispatches() const {return impl_->interleavedBodyDispatches;}'
 source=replace(source,needle,needle+'\nsize_t MetalContext::LastBlockPackedPrefillDispatches() const {return impl_->blockPackedPrefillDispatches;}\nsize_t MetalBF16Weight::BlockPackedPrefillBytes() const {return impl_&&impl_->blockPackedPrefill?impl_->blockPackedPrefill.length:0;}')
 needle='        if(!w->visionPrefill&&a.rows==160)++impl_->wideBatchPrefillDispatches;'
 source=replace(source,needle,needle+'''
        if(gBlockPackedPrefillPolicy&&a.rows>2&&w->blockPackedPrefill){
            if(gBlockPackedPrefillPolicy<1||gBlockPackedPrefillPolicy>2)throw std::invalid_argument("Block packed policy invalid");
            const size_t tileRows=gBlockPackedPrefillPolicy==1?32:64;
            auto pipeline=gBlockPackedPrefillPolicy==1?impl_->kernels->blockPackedPrefill32:impl_->kernels->blockPackedPrefill64;
            if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<256||pipeline.staticThreadgroupMemoryLength>impl_->device.maxThreadgroupMemoryLength)throw std::runtime_error("Block packed geometry unsupported");
            struct Params{uint64_t rows,inner,columns,aStride,oStride;};Params p{a.rows,a.columns,o.columns,a.rowBytes/4,o.rowBytes/4};
            auto encoder=[impl_->command computeCommandEncoder];if(!encoder)throw std::runtime_error("Block packed encoder unavailable");
            [encoder setComputePipelineState:pipeline];[encoder setBuffer:a.data offset:0 atIndex:0];[encoder setBuffer:w->blockPackedPrefill offset:0 atIndex:1];[encoder setBuffer:o.data offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];
            [encoder dispatchThreadgroups:MTLSizeMake((o.columns+63)/64,(a.rows+tileRows-1)/tileRows,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];[encoder endEncoding];
            output.impl_->ready=false;output.impl_->pending=true;if(impl_->pendingOutputs.insert(output.impl_.get()).second)impl_->outputs.push_back(output.impl_);
            ++impl_->blockPackedPrefillDispatches;return;
        }
''')
 return replace(source,'impl_->mlpGraphDispatches=0;','impl_->mlpGraphDispatches=0;impl_->blockPackedPrefillDispatches=0;',2)

def harness(source):
 source=replace(source,'gPackedBodyPrefillPolicy','gBlockPackedPrefillPolicy',2)
 source=replace(source,'LastPackedBodyPrefillDispatches','LastBlockPackedPrefillDispatches')
 source=replace(source,'need(argc==3,"Expected assets and model");','need(argc==5,"Expected assets, model, mode and timing policy");bool timing=std::strcmp(argv[3],"timing")==0;need(timing||std::strcmp(argv[3],"numeric")==0,"Unknown mode");int timingPolicy=std::atoi(argv[4]);need(!timing||(timingPolicy>=1&&timingPolicy<=2),"Bad timing policy");')
 source=replace(source,'p.weight=gpu.UploadBF16','auto packStart=std::chrono::steady_clock::now();p.weight=gpu.UploadBF16')
 source=replace(source,'p.selected=p.weight.InterleavedBodyReady();projections.push_back(std::move(p));','p.selected=p.weight.BlockPackedPrefillBytes()>0;need(p.selected==p.weight.InterleavedBodyReady(),"Block packed selection changed");emit(@{@"kind":@"packing",@"matrix":c[@"name"],@"selected":@(p.selected),@"additional_bytes":@(p.weight.BlockPackedPrefillBytes()),@"upload_ms":@(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-packStart).count())});projections.push_back(std::move(p));')
 source=replace(source,'for(int pattern=0;pattern<4;++pattern)','if(!timing)for(int pattern=0;pattern<4;++pattern)')
 source=replace(source,'for(int policy=1;policy<=3;++policy)','for(int policy=1;policy<=2;++policy)',2)
 start=source.index('        inputs(0,false);auto warm=');end=source.index('\n    }return 0;',start)
 source=source[:start]+'''        if(!timing)continue;inputs(0,false);auto warm=std::chrono::steady_clock::now();
        do{run(0);run(timingPolicy);}while(std::chrono::duration<double>(std::chrono::steady_clock::now()-warm).count()<2.);
        for(bool null:{true,false})for(int round=0;round<10;++round)for(int position=0;position<4;++position){bool candidate=position==1||position==2;int actual=candidate&&!null?timingPolicy:0;
            for(int w=0;w<3;++w)run(actual);
            for(int sample=0;sample<5;++sample){auto t=run(actual);need(std::isfinite(t.first)&&std::isfinite(t.second)&&t.first>0&&t.second>0,"Bad interval");emit(@{@"kind":@"timing",@"policy":@(timingPolicy),@"batch":@(batch),@"rows":@(rows),@"null":@(null),@"round":@(round),@"position":@(position),@"sample":@(sample),@"candidate":@(candidate),@"actual_policy":@(actual),@"gpu_ms":@(t.first),@"wall_ms":@(t.second)});}
        }'''+source[end:]
 return replace(source,'    }return 0;','    }emit(@{@"kind":@"complete",@"mode":timing?@"timing":@"numeric",@"numeric_checks":@(timing?0:16),@"timing_samples":@(timing?800:0),@"full_goal_met":@NO});return 0;')
