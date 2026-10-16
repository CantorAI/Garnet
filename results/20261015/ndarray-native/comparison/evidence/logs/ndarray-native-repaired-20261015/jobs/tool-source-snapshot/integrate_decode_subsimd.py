"""Insert the numerically checked component behind a default-OFF build option."""
from pathlib import Path
import hashlib,sys
root=Path(sys.argv[1]).resolve();shader=Path(sys.argv[2]).read_text()
path=root/'src/backends/metal/metal_context.mm';raw=path.read_bytes();source=raw.decode().replace('\r\n','\n')
assert 'GARNET_METAL_SUBSIMD_DECODE' not in source
start=shader.index('kernel void bf16_decode_s16_t64(');end=shader.index('kernel void bf16_decode_s16_t128(',start)
kernel=shader[start:end];control=source[source.index('kernel void bf16_decode('):source.index('kernel void bf16_decode_small(')]
def replace(old,new):
    global source
    assert source.count(old)==1,old;source=source.replace(old,new)
replace('    id<MTLComputePipelineState> bf16Decode,bf16DecodeSmall,bf16Gemm,bf16Gemm16Wide,bf16Gemm32Wide;',
'''    id<MTLComputePipelineState> bf16Decode,bf16DecodeSmall,bf16Gemm,bf16Gemm16Wide,bf16Gemm32Wide;
#ifdef GARNET_METAL_SUBSIMD_DECODE
    id<MTLComputePipelineState> bf16DecodeSubSIMD;
#endif''')
needle='        MTLCompileOptions* compileOptions=nil;\n'
replace(needle,needle+'''#ifdef GARNET_METAL_SUBSIMD_DECODE
        source=[source stringByAppendingString:[NSString stringWithUTF8String:R"METAL(
'''+kernel+''')METAL"]];
#endif
''')
needle='        impl_->kernels->bf16Decode=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_decode"] error:&error];'
replace(needle,needle+'''
#ifdef GARNET_METAL_SUBSIMD_DECODE
        impl_->kernels->bf16DecodeSubSIMD=[impl_->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"bf16_decode_s16_t64"] error:&error];
        if(!impl_->kernels->bf16DecodeSubSIMD||impl_->kernels->bf16DecodeSubSIMD.threadExecutionWidth!=32||impl_->kernels->bf16DecodeSubSIMD.maxTotalThreadsPerThreadgroup<64)
            throw std::runtime_error("Sub-SIMD decode requires a 32-lane pipeline with 64 threads");
#endif''')
replace('        size_t decodeColumns=16;\n','''        size_t decodeColumns=16;
#ifdef GARNET_METAL_SUBSIMD_DECODE
        if(a.rows<=2&&o.columns>256) {
            pipeline=impl_->kernels->bf16DecodeSubSIMD;threads=64;
        }
#endif
''')
assert source[source.index('kernel void bf16_decode('):source.index('kernel void bf16_decode_small(')]==control
assert kernel in source
path.write_bytes(source.encode())
print('Installed kernel SHA256',hashlib.sha256(kernel.encode()).hexdigest())
