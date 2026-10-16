"""Specialize all fixed dimensions; separately test removing proven-absent tails."""
from pathlib import Path
import sys
native=Path(sys.argv[1]).read_text();folder=Path(__file__).parent
original=('#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params {ulong rows,inner,columns,aStride,oStride;};\n'+native[native.index('kernel void bf16_decode('):native.index('kernel void bf16_gemm(')])
special=original[original.index('kernel void bf16_decode('):].replace('kernel void bf16_decode(', 'kernel void specialized_bf16_decode(').replace('kernel void bf16_decode_small(', 'kernel void specialized_bf16_decode_small(')
for name,index in (('rows',0),('inner',1),('columns',2),('aStride',3),('oStride',4)):
    special=special.replace('p.'+name,'ulong(fixed_'+name+')')
special=special.replace('if(column>=ulong(fixed_columns))return;','if(!fixed_no_tail&&column>=ulong(fixed_columns))return;')
for j in (1,2,3):special=special.replace('if(column+'+str(j)+'<ulong(fixed_columns))','if(fixed_no_tail||column+'+str(j)+'<ulong(fixed_columns))')
special=special.replace('j<4&&column+j<ulong(fixed_columns)','j<4&&(fixed_no_tail||column+j<ulong(fixed_columns))').replace('j<2&&column+j<ulong(fixed_columns)','j<2&&(fixed_no_tail||column+j<ulong(fixed_columns))')
decl=''.join('constant uint fixed_'+n+' [[function_constant('+str(i)+')]];\n' for n,i in (('rows',0),('inner',1),('columns',2),('aStride',3),('oStride',4)))+'constant bool fixed_no_tail [[function_constant(5)]];\n'
Path(sys.argv[2]).write_bytes((original+decl+special).encode())
cpp=(folder/'indirect_bf16_projection.mm').read_text()
start=cpp.index(' id<MTLComputePipelineState> recordable[2]=');end=cpp.index(' auto folder=',start)
cpp=cpp[:start]+''' emit(@{@"kind":@"device",@"device":device.name,@"specialization":@"dimensions, strides and optional proven-absent tails; original math"});
'''+cpp[end:]
cpp=cpp.replace('normal[small].threadExecutionWidth==32&&recordable[small].threadExecutionWidth==32','normal[small].threadExecutionWidth==32')
start=cpp.index('   auto descriptor=[MTLIndirectCommandBufferDescriptor new];');end=cpp.index('   std::vector<double> exact(',start)
cpp=cpp[:start]+'''   auto setupBegin=Clock::now();id<MTLComputePipelineState> specialized[2];
   require(rows<=UINT32_MAX&&inner<=UINT32_MAX&&columns<=UINT32_MAX&&aStride<=UINT32_MAX&&oStride<=UINT32_MAX,"Specialization dimension overflow");
   uint32_t dimensions[]={uint32_t(rows),uint32_t(inner),uint32_t(columns),uint32_t(aStride),uint32_t(oStride)};
   for(int mode=0;mode<2;++mode){auto constants=[MTLFunctionConstantValues new];
    for(NSUInteger i=0;i<5;++i)[constants setConstantValue:&dimensions[i] type:MTLDataTypeUInt atIndex:i];
    bool noTail=mode&&columns%tile==0;[constants setConstantValue:&noTail type:MTLDataTypeBool atIndex:5];
    auto function=[library newFunctionWithName:small?@"specialized_bf16_decode_small":@"specialized_bf16_decode" constantValues:constants error:&error];if(!function)throw std::runtime_error(error.localizedDescription.UTF8String);
    specialized[mode]=[device newComputePipelineStateWithFunction:function error:&error];if(!specialized[mode])throw std::runtime_error(error.localizedDescription.UTF8String);
    require(specialized[mode].threadExecutionWidth==32&&specialized[mode].maxTotalThreadsPerThreadgroup>=threads,"Specialized pipeline limits changed");
   }
   emit(@{@"kind":@"setup",@"matrix":name,@"rows":@(rows),@"pipeline_specialization_ms":@(elapsed(setupBegin,Clock::now())),@"tail_checks_removed":@(columns%tile==0)});
'''+cpp[end:]
start=cpp.index('    if(mode==2){');end=cpp.index('    [encoder endEncoding];',start)
cpp=cpp[:start]+'''    [encoder setComputePipelineState:mode==0?normal[small]:specialized[mode-1]];[encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:weight offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];[encoder dispatchThreadgroups:MTLSizeMake((columns+tile-1)/tile,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
'''+cpp[end:]
needle='   for(int pair=0;pair<2;++pair)';assert cpp.count(needle)==1
cpp=cpp.replace(needle,'''   size_t commonWarm=0;auto warmBegin=Clock::now();
   do{int mode=int(commonWarm%3);run(mode,int(epoch++%4),true);++commonWarm;}while(elapsed(warmBegin,Clock::now())<250);
   emit(@{@"kind":@"warmup",@"matrix":name,@"rows":@(rows),@"elapsed_ms":@(elapsed(warmBegin,Clock::now()))});
'''+needle)
cpp=cpp.replace('@"direct_indirect_flag"','@"dimensions"').replace('@"validated_executions":@240','@"validated_executions":@(240+commonWarm)')
Path(sys.argv[3]).write_bytes(cpp.encode())
print('Original shader and fixed-dimension candidates generated; bit-parity and FP64 gates retained')
