"""Use current FP32 MPS vision weights as control, exact half alpha as candidate."""
from pathlib import Path
here=Path(__file__).resolve().parent
source=(here/'global_half_alpha_gpu.mm').read_text()
source=source.replace('// Original row-scaled MPS + rescale versus exact common scale in MPS alpha.', '// Current expanded FP32 MPS vision products versus exact common half-storage alpha.')
start=source.index(' auto source=[NSString stringWithUTF8String:');end=source.index(' auto manifest=',start)
source=source[:start]+source[end:]
start=source.index('  auto rowWeight=');end=source.index('  for(NSNumber* rowValue',start)
source=source[:start]+'''  auto floatWeight=matrix(device,columns,inner,MPSDataTypeFloat32),globalWeight=matrix(device,columns,inner,MPSDataTypeFloat16);
  for(uint64_t col=0;col<columns;++col)for(uint64_t k=0;k<inner;++k){
   static_cast<float*>(floatWeight.buffer.contents)[col*floatWeight.stride+k]=learned[col*inner+k];
   static_cast<uint16_t*>(globalWeight.buffer.contents)[col*globalWeight.stride+k]=globalBits[col*inner+k];
   __fp16 half;std::memcpy(&half,&globalBits[col*inner+k],2);float restored=std::ldexp(float(half),-globalScale);
   need(std::memcmp(&restored,&learned[col*inner+k],4)==0,"Vision half storage changed learned bits");
  }
''' + source[end:]
source=source.replace(',temporary=matrix(device,rows,columns,MPSDataTypeFloat32)','')
source=source.replace(';Params p{rows,columns,temporary.stride,output.stride};',';')
source=source.replace('manifest[@"rows"]','item[@"rows"]')
start=source.index('    auto start=Clock::now();');end=source.index('    auto encoded=Clock::now();',start)
source=source[:start]+'''    auto start=Clock::now();auto command=[queue commandBuffer];
    [(useCandidate?candidate:control) encodeToCommandBuffer:command leftMatrix:input.view rightMatrix:(useCandidate?globalWeight:floatWeight).view resultMatrix:output.view];
''' + source[end:]
source=source.replace('global_half_alpha_failed:', 'vision_global_alpha_failed:')
(here/'vision_global_alpha_gpu.mm').write_bytes(source.encode())
print('Prepared current FP32 versus exact half-alpha vision product gate')
