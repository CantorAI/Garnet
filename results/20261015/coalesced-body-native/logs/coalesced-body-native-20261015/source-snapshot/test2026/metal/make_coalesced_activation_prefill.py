"""Compare a coalesced lossless-input MMA with the selected global-alpha MPS path."""
from pathlib import Path
import sys
folder=Path(__file__).parent
wrapper=(folder/'make_coalesced_activation_decode.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.index('Path(sys.argv[1])')],__file__,'exec'),ns)
cpp=ns['cpp']
start=cpp.index('  for(uint64_t col=0;col<n&&eligible;++col)')
end=cpp.index('  if(!eligible)',start)
cpp=cpp[:start]+'''  uint16_t largest=0;for(uint64_t j=0;j<k*n;++j)largest=std::max(largest,uint16_t(raw[j]&0x7fff));
  int32_t commonScale=0;eligible=Garnet::ExactHalfRowScale(largest,commonScale);
  for(uint64_t col=0;col<n&&eligible;++col){colScale[col]=commonScale;
   for(uint64_t j=0;j<k&&eligible;++j){uint16_t encoded;eligible=Garnet::EncodeExactScaledHalf(raw[col*k+j],commonScale,encoded);if(!eligible)break;halfBits[col*wStride+j]=encoded;
    __fp16 h;std::memcpy(&h,&encoded,2);float restored=std::ldexp(float(h),-commonScale);uint32_t bits;std::memcpy(&bits,&restored,4);need(bits==(uint32_t(raw[col*k+j])<<16),"Learned value changed");}
  }
'''+cpp[end:]
cpp=cpp.replace('Exact half weight admission failed; original native fallback required','Global exact-half admission failed; selected fallback remains required')
cpp=cpp.replace('for(uint64_t rows:{1,2})','for(uint64_t rows:{8,16,32,80})')
needle='   auto product=[[MPSMatrixMultiplication alloc]';pos=cpp.index(needle)
cpp=cpp[:pos]+'''   auto controlInput=[[MPSMatrix alloc] initWithBuffer:a descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:k rowBytes:aStride*4 dataType:MPSDataTypeFloat32]];
   auto controlOutput=[[MPSMatrix alloc] initWithBuffer:out descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:rows columns:n rowBytes:oStride*4 dataType:MPSDataTypeFloat32]];
   auto controlProduct=[[MPSMatrixMultiplication alloc] initWithDevice:d transposeLeft:NO transposeRight:YES resultRows:rows resultColumns:n interiorColumns:k alpha:std::ldexp(1.0,-commonScale) beta:0];
   need(controlInput&&controlOutput&&controlProduct,"Selected global-alpha MPS setup failed");
'''+cpp[pos:]
start=cpp.index('    }else{auto e=[cb computeCommandEncoder]')
end=cpp.index('    auto encoded=Clock::now()',start)
cpp=cpp[:start]+'''    }else{[controlProduct encodeToCommandBuffer:cb leftMatrix:controlInput rightMatrix:wMatrix resultMatrix:controlOutput];}
'''+cpp[end:]
cpp=cpp.replace('@"Original BF16 shader/default compile options"','@"Selected global-alpha MPS: FP32 input, exact half weights, FP32 output"')
cpp=cpp.replace('original native fallback required','selected prefill fallback required')
Path(sys.argv[1]).write_bytes(cpp.encode())
Path(sys.argv[2]).write_bytes((folder/'joined_activation_decode.metal').read_bytes()+b'\n'+(folder/'coalesced_activation_product.metal').read_bytes())
print('Prefill MMA and selected global-alpha MPS control generated; original FP64 and reconstruction bounds retained')
