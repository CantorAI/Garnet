"""Six-bit derived screening index with aligned packed-word reads."""
from pathlib import Path
import runpy
def generate_shader(source):
    source=source.replace('device const char* approximate','device const uint* approximate')
    begin=source.index('        char4 b0=');end=source.index('\n    }',begin)
    source=source[:begin]+'''        uint b0=screen_six(approximate,column*p.inner+k),b1=0,b2=0,b3=0;
        if(valid.y)b1=screen_six(approximate,(column+1)*p.inner+k);
        if(valid.z)b2=screen_six(approximate,(column+2)*p.inner+k);
        if(valid.w)b3=screen_six(approximate,(column+3)*p.inner+k);
        for(uint j=0;j<4;++j){
            float4 w=float4(int((b0>>(j*6))&63)-32,int((b1>>(j*6))&63)-32,int((b2>>(j*6))&63)-32,int((b3>>(j*6))&63)-32);
            sum0+=x[j]*w;if(p.rows==2)sum1+=y[j]*w;
        }'''+source[end:]
    return '''inline uint screen_six(device const uint* packed,ulong valueOffset) {
    ulong bit=valueOffset*6;uint shift=uint(bit%32);uint result=packed[bit/32]>>shift;
    if(shift>8)result|=packed[bit/32+1]<<((32-shift)&31);
    return result;
}
kernel void screen_index_six_roundtrip(device const uint* packed [[buffer(0)]],device char* decoded [[buffer(1)]],
    constant ulong& count [[buffer(2)]],uint id [[thread_position_in_grid]]) {
    if(ulong(id)>=count)return;
    uint values=screen_six(packed,ulong(id)&~ulong(3));decoded[id]=char(int((values>>((id%4)*6))&63)-32);
}
'''+source
def generate_harness(source):
    generate=runpy.run_path(str(Path(__file__).with_name('make_screened_head_harness.py')))['generate']
    source=generate(source)
    source=source.replace('approximate.length!=columns*inner)','approximate.length!=columns*inner*3/4)').replace('double((columns-seedRows)*inner+2*reads)','double((columns-seedRows)*inner*3/4+2*reads)')
    needle='        for(NSDictionary* query in manifest[@"queries"]) {';assert source.count(needle)==1
    source=source.replace(needle,'''        {
            auto unpack=pipeline(device,library,@"screen_index_six_roundtrip");uint64_t count=columns*inner;
            auto decoded=[device newBufferWithLength:count options:MTLResourceStorageModeShared];if(!decoded)throw std::runtime_error("Index validation allocation failed");
            auto cb=[queue commandBuffer];auto enc=[cb computeCommandEncoder];[enc setComputePipelineState:unpack];
            [enc setBuffer:approximate offset:0 atIndex:0];[enc setBuffer:decoded offset:0 atIndex:1];[enc setBytes:&count length:sizeof(count) atIndex:2];
            [enc dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[enc endEncoding];finish(cb);
            auto actual=static_cast<const int8_t*>(decoded.contents);auto original=static_cast<const uint16_t*>(weight.contents);auto m=static_cast<const float*>(remaining.contents);
            for(uint64_t i=0;i<count;++i){uint32_t bits=uint32_t(original[i])<<16;float w;std::memcpy(&w,&bits,4);
                if(int(actual[i])!=int(std::nearbyint(double(w)/double(m[(i/inner)*4+3]))))throw std::runtime_error("Complete GPU six-bit reconstruction mismatch");
            }
            printf("{\\"kind\\":\\"index_roundtrip\\",\\"bits\\":6,\\"elements\\":%llu,\\"all_values_exact\\":true}\\n",count);fflush(stdout);
        }
'''+needle)
    return source
