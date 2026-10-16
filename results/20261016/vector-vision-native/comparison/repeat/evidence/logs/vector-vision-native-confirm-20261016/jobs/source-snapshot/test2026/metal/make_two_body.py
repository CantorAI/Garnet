"""Two outputs per SIMD group over the unchanged four-output compact payload."""
from make_coalesced_eight_body import context as base_context,header as base_header,harness as base_harness

def context(source):
    result=base_context(source)
    assert result.count('decodeColumns=threads/32*8;')==1
    result=result.replace('decodeColumns=threads/32*8;','decodeColumns=threads/32*2;')
    for old,new in (
        ('coalesced_eight_body_shader.h','coalesced_two_body_shader.h'),
        ('kCoalescedEightBodyShader','kCoalescedTwoBodyShader'),
        ('bf16_body_coalesced_eight','bf16_body_coalesced_two'),
        ('bf16BodyCoalescedEight','bf16BodyCoalescedTwo'),
        ('eightBodyDispatches','twoBodyDispatches'),
        ('LastEightBodyDispatches','LastTwoBodyDispatches')):
        assert old in result,old
        result=result.replace(old,new)
    return result

def header(source):return base_header(source).replace('LastEightBodyDispatches','LastTwoBodyDispatches')
def harness(source):return base_harness(source).replace('LastEightBodyDispatches','LastTwoBodyDispatches').replace('Eight body','Two body').replace('Eight safety','Two safety').replace('Eight abort','Two abort')

def shader(source):
    def replace(old,new):
        nonlocal source
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new)
    replace('ulong column=ulong(group)*(threads/32)*4+simd*4;',
            'ulong column=ulong(group)*(threads/32)*2+simd*2;')
    replace('float4 sum0=0.0f,sum1=0.0f;','float2 sum0=0.0f,sum1=0.0f;')
    replace('uint meta=metadata[block];float4 weights[4];','uint meta=metadata[block];float2 weights[4];')
    replace('uint4 signMant=uint4(as_type<uchar4>(sm[j]));',
            'uint2 signMant=uint2(as_type<uchar2>(ushort(sm[j]>>uint((column%4)*8))));')
    replace('uint4 exponents=uint4(tags[j],tags[j]>>4,tags[j]>>8,tags[j]>>12)&15u;',
            'uint shift=uint((column%4)*4);uint2 exponents=uint2(tags[j]>>shift,tags[j]>>(shift+4))&15u;')
    replace('uint4 bits=(signMant&127u)|((signMant&128u)<<8)|((exponents+(meta&255u))<<7);',
            'uint2 bits=(signMant&127u)|((signMant&128u)<<8)|((exponents+(meta&255u))<<7);')
    replace('weights[j]=as_type<float4>(bits<<16);','weights[j]=as_type<float2>(bits<<16);')
    replace('for(uint j=0;j<4;++j)weights[j]=as_type<float4>(uint4(words[j])<<16);',
            'for(uint j=0;j<4;++j)weights[j]=as_type<float2>(uint2((column%4)?words[j].zw:words[j].xy)<<16);')
    replace('sum0+=x[j]*weights[j];if(p.rows==2)sum1+=y[j]*weights[j];',
            'sum0=fma(float2(x[j]),weights[j],sum0);if(p.rows==2)sum1=fma(float2(y[j]),weights[j],sum1);')
    for name in ('sum0','sum1'):
        replace(name+'=float4(simd_sum('+name+'.x),simd_sum('+name+'.y),simd_sum('+name+'.z),simd_sum('+name+'.w));',
                name+'=float2(simd_sum('+name+'.x),simd_sum('+name+'.y));')
    replace('j<4&&column+j<p.columns','j<2&&column+j<p.columns')
    return source.replace('kCoalescedBodyBlockShader','kCoalescedTwoBodyShader').replace('bf16_body_coalesced_block(','bf16_body_coalesced_two(').replace(
        '// Private exact BF16 block decode: four outputs by128 K, coalesced byte planes.',
        '// Private two-output compute; original four-output compact payload unchanged.')
