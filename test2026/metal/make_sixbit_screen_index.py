"""Six-bit follow-up to excessive four-bit survivors; same original score/error predicates."""
from make_nibble_screen_index import replace,header,context,index as nibble_index,native_impl as nibble_impl

def index(source):
 source=nibble_index(source)
 return replace(source,'maximum!=7&&maximum!=127','maximum!=31&&maximum!=127')

def native_impl(source):
 source=nibble_impl(source)
 source=replace(source,'nibble?7:127','nibble?31:127')
 source=replace(source,'Elements(paddedRows,w->columns)/(nibble?2:1)','Elements(paddedRows,w->columns)*(nibble?3:4)/4')
 source=replace(source,'if(nibble)destination[at/2]|=(uint8_t(q)&15u)<<((at%2)*4);else destination[at]=uint8_t(q);','if(nibble){size_t base=(group*w->columns+k)*3;uint32_t bits=uint32_t(uint8_t(q)&63u)<<(6*c);for(size_t byte=0;byte<3;++byte)destination[base+byte]|=uint8_t(bits>>(8*byte));}else destination[at]=uint8_t(q);')
 return replace(source,'if(nibble){value=(destination[at/2]>>((at%2)*4))&15u;if(value>=8)value-=16;}','if(nibble){size_t base=((row/4)*w->columns+k)*3;uint32_t bits=uint32_t(destination[base])|(uint32_t(destination[base+1])<<8)|(uint32_t(destination[base+2])<<16);value=(bits>>(6*(row%4)))&63u;if(value>=32)value-=64;}')

def shader(source):
 source=replace(source,'kScreenedHeadShader','kNibbleScreenedHeadShader')
 source=replace(source,'prepare_head_bounds','nibble_prepare_head_bounds')
 source=replace(source,'kernel void screened_head(','kernel void nibble_screened_head(')
 old='''        uint4 packed=*(device const uint4*)(approximate+column*p.inner+k*4);
        for(uint j=0;j<4;++j){float4 w=float4(as_type<char4>(packed[j]));sum0=fma(x[j],w,sum0);if(p.rows==2)sum1=fma(y[j],w,sum1);}'''
 new='''        uint3 words=uint3(*(device const packed_uint3*)(approximate+(column*p.inner+k*4)*3/4));
        uint4 packed=uint4(words.x,(words.x>>24)|(words.y<<8),(words.y>>16)|(words.z<<16),words.z>>8);
        for(uint j=0;j<4;++j){uint4 tags=uint4(packed[j],packed[j]>>6,packed[j]>>12,packed[j]>>18)&63u;
            int4 signedValues=int4(tags)-select(int4(0),int4(64),tags>=32u);float4 w=float4(signedValues);
            sum0=fma(x[j],w,sum0);if(p.rows==2)sum1=fma(y[j],w,sum1);}'''
 return replace(source,old,new)
