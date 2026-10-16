"""Replace only the rejected MPS component product; retain strict exactness gates."""
from pathlib import Path
import sys
folder=Path(__file__).parent;cpp=(folder/'joined_activation_decode_gpu.mm').read_text()
cpp=cpp.replace('One MPS half product with lossless three-component input and FP32 result/recombination','Coalesced half components/weights with explicit FP32 SIMD accumulator and FP32 recombination')
needle='auto split=pipeline(d,candidateLibrary,@"joined_split"),combine=pipeline(d,candidateLibrary,@"joined_combine");';assert cpp.count(needle)==1
cpp=cpp.replace(needle,needle+'''auto mixed=pipeline(d,candidateLibrary,@"coalesced_product");
 need(mixed.threadExecutionWidth==32&&mixed.maxTotalThreadsPerThreadgroup>=128&&mixed.staticThreadgroupMemoryLength<=d.maxThreadgroupMemoryLength,"Mixed matrix pipeline exceeds device limits");''')
needle='     [product encodeToCommandBuffer:cb leftMatrix:left rightMatrix:wMatrix resultMatrix:result];';assert cpp.count(needle)==1
cpp=cpp.replace(needle,'''     struct MixedP{uint64_t rows,inner,columns,leftStride,weightStride,outputStride;};MixedP mp{rows*3,k,n,hStride,wStride,oStride};
     e=[cb computeCommandEncoder];need(e!=nil,"Mixed matrix encoder unavailable");[e setComputePipelineState:mixed];[e setBuffer:joined offset:0 atIndex:0];[e setBuffer:halfWeight offset:0 atIndex:1];[e setBuffer:parts offset:0 atIndex:2];[e setBytes:&mp length:sizeof(mp) atIndex:3];[e dispatchThreadgroups:MTLSizeMake((n+31)/32,(rows*3+7)/8,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];''')
assert 'product encodeToCommandBuffer' not in cpp
Path(sys.argv[1]).write_bytes(cpp.encode())
Path(sys.argv[2]).write_bytes((folder/'joined_activation_decode.metal').read_bytes()+b'\n'+(folder/'coalesced_activation_product.metal').read_bytes())
print('Coalesced product generated; original admission/reconstruction/FP64/control gates retained')
