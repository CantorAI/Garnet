static constexpr const char* kBlockPackedPrefillShader=R"BLOCK(kernel void block_packed_prefill_32(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float scratch[2048],right[2048];
ulong rowBase=ulong(group.y)*32,colBase=ulong(group.x)*64,kTiles=(p.inner+31)/32;
simdgroup_float8x8 acc0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 acc1=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 acc2=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 acc3=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=32){
for(uint i=tid;i<1024;i+=256){ulong row=rowBase+i/32,k=base+i%32;scratch[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<2048;i+=256)right[i]=as_type<float>(uint(weight[(ulong(group.x)*kTiles+base/32)*2048+i])<<16);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<32;offset+=8){simdgroup_float8x8 r;simdgroup_load(r,right+offset*64+simd*8,64);
if(rowBase+0<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+0+offset,32);simdgroup_multiply_accumulate(acc0,l,r,acc0);}
if(rowBase+8<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+256+offset,32);simdgroup_multiply_accumulate(acc1,l,r,acc1);}
if(rowBase+16<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+512+offset,32);simdgroup_multiply_accumulate(acc2,l,r,acc2);}
if(rowBase+24<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+768+offset,32);simdgroup_multiply_accumulate(acc3,l,r,acc3);}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
if(rowBase+0<p.rows)simdgroup_store(acc0,scratch+0+simd*8,64);
if(rowBase+8<p.rows)simdgroup_store(acc1,scratch+512+simd*8,64);
if(rowBase+16<p.rows)simdgroup_store(acc2,scratch+1024+simd*8,64);
if(rowBase+24<p.rows)simdgroup_store(acc3,scratch+1536+simd*8,64);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<2048;i+=256){ulong row=rowBase+i/64,column=colBase+i%64;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=scratch[i];}
}
kernel void block_packed_prefill_64(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
threadgroup float scratch[4096],right[2048];
ulong rowBase=ulong(group.y)*64,colBase=ulong(group.x)*64,kTiles=(p.inner+31)/32;
simdgroup_float8x8 acc0=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 acc1=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 acc2=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 acc3=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 acc4=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 acc5=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 acc6=make_filled_simdgroup_matrix<float,8>(0.0f);
simdgroup_float8x8 acc7=make_filled_simdgroup_matrix<float,8>(0.0f);
for(ulong base=0;base<p.inner;base+=32){
for(uint i=tid;i<2048;i+=256){ulong row=rowBase+i/32,k=base+i%32;scratch[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}
for(uint i=tid;i<2048;i+=256)right[i]=as_type<float>(uint(weight[(ulong(group.x)*kTiles+base/32)*2048+i])<<16);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint offset=0;offset<32;offset+=8){simdgroup_float8x8 r;simdgroup_load(r,right+offset*64+simd*8,64);
if(rowBase+0<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+0+offset,32);simdgroup_multiply_accumulate(acc0,l,r,acc0);}
if(rowBase+8<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+256+offset,32);simdgroup_multiply_accumulate(acc1,l,r,acc1);}
if(rowBase+16<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+512+offset,32);simdgroup_multiply_accumulate(acc2,l,r,acc2);}
if(rowBase+24<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+768+offset,32);simdgroup_multiply_accumulate(acc3,l,r,acc3);}
if(rowBase+32<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+1024+offset,32);simdgroup_multiply_accumulate(acc4,l,r,acc4);}
if(rowBase+40<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+1280+offset,32);simdgroup_multiply_accumulate(acc5,l,r,acc5);}
if(rowBase+48<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+1536+offset,32);simdgroup_multiply_accumulate(acc6,l,r,acc6);}
if(rowBase+56<p.rows){simdgroup_float8x8 l;simdgroup_load(l,scratch+1792+offset,32);simdgroup_multiply_accumulate(acc7,l,r,acc7);}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
}
if(rowBase+0<p.rows)simdgroup_store(acc0,scratch+0+simd*8,64);
if(rowBase+8<p.rows)simdgroup_store(acc1,scratch+512+simd*8,64);
if(rowBase+16<p.rows)simdgroup_store(acc2,scratch+1024+simd*8,64);
if(rowBase+24<p.rows)simdgroup_store(acc3,scratch+1536+simd*8,64);
if(rowBase+32<p.rows)simdgroup_store(acc4,scratch+2048+simd*8,64);
if(rowBase+40<p.rows)simdgroup_store(acc5,scratch+2560+simd*8,64);
if(rowBase+48<p.rows)simdgroup_store(acc6,scratch+3072+simd*8,64);
if(rowBase+56<p.rows)simdgroup_store(acc7,scratch+3584+simd*8,64);
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint i=tid;i<4096;i+=256){ulong row=rowBase+i/64,column=colBase+i%64;if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=scratch[i];}
}
)BLOCK";
