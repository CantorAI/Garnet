"""Exact INT8 values in FP16 storage for MPS filtering; original BF16 refinement."""
from make_screened_seed_abba import generate_harness as base_harness
from make_screened_seed_probe import generate_shader as seed_shader
def generate_shader(source):
    control=seed_shader(source,1024).replace('prepare_head_bounds','control_prepare_head_bounds').replace('screened_head(','control_screened_head(')
    candidate=seed_shader(source,1024)
    start=candidate.index('    float4 sum0=0,sum1=0;');end=candidate.index('    bool4 keep0=valid',start)
    candidate=candidate[:start]+'''    float4 sum0=float4(approximate[column],valid.y?approximate[column+1]:0,valid.z?approximate[column+2]:0,valid.w?approximate[column+3]:0);
    float4 sum1=p.rows==2?float4(approximate[p.columns+column],valid.y?approximate[p.columns+column+1]:0,valid.z?approximate[p.columns+column+2]:0,valid.w?approximate[p.columns+column+3]:0):float4(0);
'''+candidate[end:]
    candidate=candidate.replace('device const char* approximate','device const float* approximate')
    return candidate+'\n'+control
def generate_harness(source):
    source=base_harness(source)
    def change(old,new):
        nonlocal source
        assert source.count(old)==1,old
        source=source.replace(old,new)
    source=source.replace('#import <Metal/Metal.h>','#import <Metal/Metal.h>\n#import <MetalPerformanceShaders/MetalPerformanceShaders.h>')
    change('batch*8192*4','batch*1024*4')
    change('Params controlSeedParams{batch,inner,8192,inner,8192};','Params controlSeedParams{batch,inner,1024,inner,1024};')
    needle='        auto metadata=static_cast<const float*>(remaining.contents);'
    change(needle,'''        auto indexHalf=[device newBufferWithLength:columns*inner*2 options:MTLResourceStorageModeShared];
        if(!indexHalf)throw std::runtime_error("MPS filter index allocation failed");
        auto originalIndex=static_cast<const int8_t*>(approximate.contents);auto packedIndex=static_cast<__fp16*>(indexHalf.contents);
        for(uint64_t i=0;i<columns*inner;++i){packedIndex[i]=__fp16(originalIndex[i]);if(float(packedIndex[i])!=float(originalIndex[i]))throw std::runtime_error("MPS filter index changed an integer");}
        printf("{\\"kind\\":\\"index_roundtrip\\",\\"elements\\":%llu,\\"all_values_exact\\":true,\\"storage\\":\\"f16-exact-int8\\"}\\n",columns*inner);fflush(stdout);
        auto indexDescriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:columns columns:inner rowBytes:inner*2 dataType:MPSDataTypeFloat16];
        auto indexMatrix=[[MPSMatrix alloc] initWithBuffer:indexHalf descriptor:indexDescriptor];
'''+needle)
    needle='            auto run=[&](bool pruned) {'
    change(needle,'''            auto dotBuffer=[device newBufferWithLength:batch*columns*4 options:MTLResourceStorageModeShared];
            if(!dotBuffer)throw std::runtime_error("MPS filter scores allocation failed");
            auto inputDescriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:batch columns:inner rowBytes:inner*4 dataType:MPSDataTypeFloat32];
            auto dotDescriptor=[MPSMatrixDescriptor matrixDescriptorWithRows:batch columns:columns rowBytes:columns*4 dataType:MPSDataTypeFloat32];
            auto inputMatrix=[[MPSMatrix alloc] initWithBuffer:a descriptor:inputDescriptor];auto dotMatrix=[[MPSMatrix alloc] initWithBuffer:dotBuffer descriptor:dotDescriptor];
            auto filterProduct=[[MPSMatrixMultiplication alloc] initWithDevice:device transposeLeft:NO transposeRight:YES resultRows:batch resultColumns:columns interiorColumns:inner alpha:1 beta:0];
            if(!indexMatrix||!inputMatrix||!dotMatrix||!filterProduct)throw std::runtime_error("MPS filter setup failed");
'''+needle)
    change('if(pruned||timingControl) {','if(pruned||timingControl) {\n                    if(pruned)[filterProduct encodeToCommandBuffer:cb leftMatrix:inputMatrix rightMatrix:indexMatrix resultMatrix:dotMatrix];')
    change('[encoder setBuffer:approximate offset:0 atIndex:8];','[encoder setBuffer:pruned?dotBuffer:approximate offset:0 atIndex:8];')
    change('double((columns-seedRows)*inner+2*reads)/double(2*columns*inner)','double(2*columns*inner+2*reads)/double(2*columns*inner)')
    source=source.replace('pruned?"seed1024":"seed8192"','pruned?"mps-filter":"int8-filter"')
    return source
