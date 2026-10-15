"""Original-kernel gate: device-readonly versus constant-address BF16 weights."""
from pathlib import Path
import sys
text=Path(__file__).with_name('indirect_bf16_projection.mm').read_text()
text=text.replace('makePipeline(device,library,@"bf16_decode",true)','makePipeline(device,library,@"constant_bf16_decode",false)').replace('makePipeline(device,library,@"bf16_decode_small",true)','makePipeline(device,library,@"constant_bf16_decode_small",false)')
text=text.replace('@"indirect_supported":@(recordable[0].supportIndirectCommandBuffers&&recordable[1].supportIndirectCommandBuffers)','@"weight_access":@"device-readonly versus constant"')
start=text.index('   auto descriptor=[MTLIndirectCommandBufferDescriptor new];');end=text.index('   std::vector<double> exact(',start)
text=text[:start]+text[end:]
start=text.index('    if(mode==2){');end=text.index('    [encoder endEncoding];',start)
text=text[:start]+'''    [encoder setComputePipelineState:mode==2?recordable[small]:normal[small]];[encoder setBuffer:a offset:0 atIndex:0];[encoder setBuffer:weight offset:0 atIndex:1];[encoder setBuffer:out offset:0 atIndex:2];[encoder setBytes:&p length:sizeof(p) atIndex:3];[encoder dispatchThreadgroups:MTLSizeMake((columns+tile-1)/tile,1,1) threadsPerThreadgroup:MTLSizeMake(threads,1,1)];
'''+text[end:]
needle='   for(int pair=0;pair<2;++pair)';assert text.count(needle)==1
text=text.replace(needle,'''   size_t commonWarm=0;auto warmBegin=Clock::now();
   do{run(commonWarm%2?2:0,int(epoch++%4),true);++commonWarm;}while(elapsed(warmBegin,Clock::now())<250);
   emit(@{@"kind":@"warmup",@"matrix":name,@"rows":@(rows),@"validated_executions":@(commonWarm),@"elapsed_ms":@(elapsed(warmBegin,Clock::now()))});
   for(int pair=0;pair<1;++pair)''')
text=text.replace('@"validated_executions":@240','@"validated_executions":@(120+commonWarm)')
Path(sys.argv[1]).write_bytes(text.encode())
