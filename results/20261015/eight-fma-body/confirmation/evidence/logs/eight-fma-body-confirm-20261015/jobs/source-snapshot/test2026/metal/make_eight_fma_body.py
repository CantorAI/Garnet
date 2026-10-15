"""Private explicit FP32 FMA experiment; original exact-bit gate still required."""
from make_coalesced_eight_body import context as base_context,header,harness

def context(source):
    result=base_context(source)
    for old,new,count in (
        ('"coalesced_eight_body_shader.h"','"coalesced_eight_fma_shader.h"',1),
        ('kCoalescedEightBodyShader','kCoalescedEightFmaBodyShader',1),
        ('@"bf16_body_coalesced_eight"','@"bf16_body_coalesced_eight_fma"',1)):
        assert result.count(old)==count
        result=result.replace(old,new)
    return result

def shader(source):
    result=source
    for pair in range(2):
        for row,activation in ((0,'x'),(1,'y')):
            accumulator=f'acc{pair}{row}'
            old=accumulator+'+='+activation+'[j]*weights[j]'
            new=accumulator+'=fma(float4('+activation+'[j]),weights[j],'+accumulator+')'
            assert result.count(old)==1,old
            result=result.replace(old,new)
    assert result.count('fma(float4(')==4
    return result.replace('kCoalescedEightBodyShader','kCoalescedEightFmaBodyShader').replace('bf16_body_coalesced_eight(','bf16_body_coalesced_eight_fma(')
