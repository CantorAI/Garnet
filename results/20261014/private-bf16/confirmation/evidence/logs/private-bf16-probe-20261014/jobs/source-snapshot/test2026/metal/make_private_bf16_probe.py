"""Adapt the exact original-kernel projection gate for private GPU weights."""
from pathlib import Path
import sys
text=Path(__file__).with_name('indirect_bf16_projection.mm').read_text()
start=text.index(' id<MTLComputePipelineState> recordable[2]=');end=text.index(' auto folder=',start)
text=text[:start]+''' emit(@{@"kind":@"device",@"device":device.name,@"small_output_policy":@"N <= 256",@"weight_storage":@"shared versus private"});
'''+text[end:]
text=text.replace('normal[small].threadExecutionWidth==32&&recordable[small].threadExecutionWidth==32','normal[small].threadExecutionWidth==32')
start=text.index('   auto descriptor=[MTLIndirectCommandBufferDescriptor new];');end=text.index('   std::vector<double> exact(',start)
text=text[:start]+'''   auto setupBegin=Clock::now();const size_t padded=(bytes.length+3)/4*4;
   auto staging=[device newBufferWithLength:padded options:MTLResourceStorageModeShared];
   auto privateWeight=[device newBufferWithLength:padded options:MTLResourceStorageModePrivate];
   auto readback=[device newBufferWithLength:padded options:MTLResourceStorageModeShared];
   require(staging&&privateWeight&&readback&&privateWeight.contents==nullptr,"Private upload buffers unavailable");
   std::memset(staging.contents,0,padded);std::memcpy(staging.contents,bytes.bytes,bytes.length);
   auto transfer=[queue commandBuffer];auto blit=[transfer blitCommandEncoder];
   [blit copyFromBuffer:staging sourceOffset:0 toBuffer:privateWeight destinationOffset:0 size:padded];
   [blit copyFromBuffer:privateWeight sourceOffset:0 toBuffer:readback destinationOffset:0 size:padded];
   [blit endEncoding];[transfer commit];[transfer waitUntilCompleted];
   require(transfer.status==MTLCommandBufferStatusCompleted&&std::memcmp(readback.contents,staging.contents,padded)==0,"Private learned bytes changed");
   emit(@{@"kind":@"setup",@"matrix":name,@"rows":@(rows),@"copy_and_readback_ms":@(elapsed(setupBegin,Clock::now())),@"private_bytes_exact":@YES});
'''+text[end:]
start=text.index('    if(mode==2){');end=text.index('    [encoder endEncoding];',start)
text=text[:start]+'''    [encoder setComputePipelineState:normal[small]];[encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:mode==2?privateWeight:weight offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];[encoder dispatchThreadgroups:MTLSizeMake((columns+tile-1)/tile,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
'''+text[end:]
needle='   for(int pair=0;pair<2;++pair)'
assert text.count(needle)==1
text=text.replace(needle,'''   size_t commonWarm=0;auto warmBegin=Clock::now();
   do{run(commonWarm%2?2:0,int(epoch++%4),true);++commonWarm;}while(elapsed(warmBegin,Clock::now())<250);
   emit(@{@"kind":@"warmup",@"matrix":name,@"rows":@(rows),@"validated_executions":@(commonWarm),@"elapsed_ms":@(elapsed(warmBegin,Clock::now()))});
   for(int pair=0;pair<1;++pair)''')
text=text.replace('@"validated_executions":@240','@"validated_executions":@(120+commonWarm)')
Path(sys.argv[1]).write_bytes(text.encode())
