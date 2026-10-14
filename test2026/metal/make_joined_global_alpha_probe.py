"""Compare two exact common-alpha MPS products with one joined product/views."""
from pathlib import Path

here=Path(__file__).resolve().parent
source=(here/'global_half_alpha_gpu.mm').read_text()
source=source.replace('// Original row-scaled MPS + rescale versus exact common scale in MPS alpha.', '// Two common-alpha MPS projections versus one joined product with output subviews.')
start=source.index(' auto source=[NSString stringWithUTF8String:')
end=source.index(' auto manifest=',start)
source=source[:start]+source[end:]
source=source.replace(',temporary=matrix(device,rows,columns,MPSDataTypeFloat32)','')
source=source.replace(';Params p{rows,columns,temporary.stride,output.stride};',';')
start=source.index('  auto rowWeight=')
end=source.index('  for(NSNumber* rowValue',start)
source=source[:start]+'''  need(columns%2==0,"Joined pair requires equal column counts");
  auto globalWeight=matrix(device,columns,inner,MPSDataTypeFloat16);
  for(uint64_t col=0;col<columns;++col)for(uint64_t k=0;k<inner;++k){
   static_cast<uint16_t*>(globalWeight.buffer.contents)[col*globalWeight.stride+k]=globalBits[col*inner+k];
   __fp16 half;std::memcpy(&half,&globalBits[col*inner+k],2);float restored=std::ldexp(float(half),-globalScale);
   need(std::memcmp(&restored,&learned[col*inner+k],4)==0,"Joined encoding changed learned bits");
  }
  auto weightDescriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:columns/2 columns:inner rowBytes:globalWeight.view.rowBytes dataType:MPSDataTypeFloat16];
  auto firstWeight=[[MPSMatrix alloc] initWithBuffer:globalWeight.buffer offset:0 descriptor:weightDescriptor];
  auto secondWeight=[[MPSMatrix alloc] initWithBuffer:globalWeight.buffer offset:(columns/2)*globalWeight.view.rowBytes descriptor:weightDescriptor];
  need(firstWeight&&secondWeight,"Weight subviews failed");
''' + source[end:]
old='resultColumns:columns interiorColumns:inner alpha:1 beta:0];'
assert source.count(old)==1
source=source.replace(old,'resultColumns:columns/2 interiorColumns:inner alpha:std::ldexp(1.0,-globalScale) beta:0];')
needle='   const float variants[4]='
assert source.count(needle)==1
source=source.replace(needle,'''   auto outputDescriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:columns/2 rowBytes:output.view.rowBytes dataType:MPSDataTypeFloat32];
   auto firstOutput=[[MPSMatrix alloc] initWithBuffer:output.buffer offset:0 descriptor:outputDescriptor];
   auto secondOutput=[[MPSMatrix alloc] initWithBuffer:output.buffer offset:(columns/2)*sizeof(float) descriptor:outputDescriptor];
   need(firstOutput&&secondOutput,"Output subviews failed");
''' + needle)
start=source.index('    auto start=Clock::now();')
end=source.index('    auto encoded=Clock::now();',start)
source=source[:start]+'''    auto start=Clock::now();auto command=[queue commandBuffer];
    if(useCandidate){[candidate encodeToCommandBuffer:command leftMatrix:input.view rightMatrix:globalWeight.view resultMatrix:output.view];}
    else {
     [control encodeToCommandBuffer:command leftMatrix:input.view rightMatrix:firstWeight resultMatrix:firstOutput];
     [control encodeToCommandBuffer:command leftMatrix:input.view rightMatrix:secondWeight resultMatrix:secondOutput];
    }
''' + source[end:]
source=source.replace('@"Common scale is not exact"','@"Joined common scale is not exact"')
source=source.replace('global_half_alpha_failed:', 'joined_global_alpha_failed:')
source=source.replace('Global alpha exceeds original comparison envelope','Joined product exceeds two-product comparison envelope')
target=here/'joined_global_alpha_gpu.mm'
target.write_bytes(source.encode())
print('Prepared',target.name)
