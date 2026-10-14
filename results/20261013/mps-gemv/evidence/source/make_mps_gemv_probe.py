"""Adapt independent FP64 matrix checks for lossless-FP32 MPS GEMV."""
from pathlib import Path
import sys
generator=Path(__file__).with_name('make_decode_subsimd_probe.py').read_text()
exec(compile(generator,str(Path(__file__).with_name('make_decode_subsimd_probe.py')),'exec'))
path=Path(sys.argv[1]);source=path.read_text()
source=source.replace('#import <Metal/Metal.h>','#import <Metal/Metal.h>\n#import <MetalPerformanceShaders/MetalPerformanceShaders.h>')
start=source.index('        std::vector<Tile> tiles=');end=source.index('        NSString* folder=',start)
source=source[:start]+'''        std::vector<Tile> tiles={{"bf16_decode",1,16,4,128,nil},{"mps_gemv_fp32",1,16,4,0,nil}};
        tiles[0].pipeline=pipeline(device,library,tiles[0].name);
        if(tiles[0].pipeline.threadExecutionWidth!=32||tiles[0].pipeline.maxTotalThreadsPerThreadgroup<128)throw std::runtime_error("Control device limit");
'''+source[end:]
needle='            for(uint64_t rows:{1,2}) {'
assert source.count(needle)==1
source=source.replace(needle,'''            const auto floatStride=[MPSMatrixDescriptor rowBytesFromColumns:inner dataType:MPSDataTypeFloat32];
            auto expanded=[device newBufferWithLength:columns*floatStride options:MTLResourceStorageModeShared];
            if(!expanded)throw std::runtime_error("Lossless FP32 allocation failed");
            std::memset(expanded.contents,0,columns*floatStride);
            auto original=static_cast<const uint16_t*>(bytes.bytes);
            for(uint64_t c=0;c<columns;++c)for(uint64_t k=0;k<inner;++k) {
                uint32_t bits=uint32_t(original[c*inner+k])<<16;
                std::memcpy(static_cast<char*>(expanded.contents)+c*floatStride+k*4,&bits,4);
            }
            auto matrix=[[MPSMatrix alloc] initWithBuffer:expanded descriptor:[MPSMatrixDescriptor matrixDescriptorWithRows:columns columns:inner rowBytes:floatStride dataType:MPSDataTypeFloat32]];
            auto gemv=[[MPSMatrixVectorMultiplication alloc] initWithDevice:device transpose:NO rows:columns columns:inner alpha:1 beta:0];
            if(!matrix||!gemv)throw std::runtime_error("MPS GEMV unavailable");
            for(uint64_t rows:{1,2}) {''')
source=source.replace('aStride=inner+8,oStride=columns+7','aStride=((inner+15)/16)*16+16,oStride=((columns+15)/16)*16+16')
needle='                auto run=[&](const Tile& tile) {'
source=source.replace(needle,'''                MPSVector* vectors[2];MPSVector* results[2];
                for(uint64_t row=0;row<rows;++row) {
                    vectors[row]=[[MPSVector alloc] initWithBuffer:a offset:row*aStride*4 descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:inner dataType:MPSDataTypeFloat32]];
                    results[row]=[[MPSVector alloc] initWithBuffer:out offset:row*oStride*4 descriptor:[MPSVectorDescriptor vectorDescriptorWithLength:columns dataType:MPSDataTypeFloat32]];
                    if(!vectors[row]||!results[row])throw std::runtime_error("MPS GEMV vector view unavailable");
                }
                auto run=[&](const Tile& tile) {''')
start=source.index('                    auto begin=std::chrono::steady_clock::now();');end=source.index('                    return std::pair<double,double>',start)
source=source[:start]+'''                    auto begin=std::chrono::steady_clock::now();auto cb=[queue commandBuffer];
                    if(tile.pipeline) {
                        auto enc=[cb computeCommandEncoder];
                        [enc setComputePipelineState:tile.pipeline];[enc setBuffer:a offset:0 atIndex:0];[enc setBuffer:weight offset:0 atIndex:1];[enc setBuffer:out offset:0 atIndex:2];[enc setBytes:&p length:sizeof(p) atIndex:3];
                        [enc dispatchThreadgroups:MTLSizeMake((columns+15)/16,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];
                    } else for(uint64_t row=0;row<rows;++row)[gemv encodeToCommandBuffer:cb inputMatrix:matrix inputVector:vectors[row] resultVector:results[row]];
                    finish(cb);
'''+source[end:]
source=source.replace('Sub-SIMD exceeds unchanged numerical bounds','MPS GEMV exceeds unchanged numerical bounds')
path.write_bytes(source.encode())
