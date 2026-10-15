// Isolated greedy-head experiment. The complete public logit API is unchanged.
// Current native bf16_decode and argmax_rows kernels precede this source.
kernel void prepare_head_bounds(device const float* input [[buffer(0)]],device const float* seed [[buffer(1)]],
    device float* stats [[buffer(2)]],constant BF16Params& p [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {
    ulong blocks=p.inner/128,seedRows=min(ulong(8192),p.columns),stride=blocks+2;
    threadgroup float partial[4],norm[64];
    float best=-INFINITY;
    for(ulong i=tid;i<seedRows;i+=128)best=max(best,seed[row*seedRows+i]);
    best=simd_max(best);if(lane==0)partial[simd]=best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(tid==0)stats[row*stride]=max(max(partial[0],partial[1]),max(partial[2],partial[3]));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(ulong block=0;block<blocks;++block) {
        float x=input[row*p.aStride+block*128+tid];float sum=simd_sum(x*x);
        if(lane==0)partial[simd]=sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if(tid==0)norm[block]=partial[0]+partial[1]+partial[2]+partial[3];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if(tid==0) {
        float tail=0;stats[row*stride+blocks+1]=0;
        for(long block=long(blocks)-1;block>=0;--block){tail+=norm[block];stats[row*stride+1+block]=tail*1.001f;}
    }
}
kernel void pruned_head(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],
    device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],device const float* stats [[buffer(4)]],
    device const float* remaining [[buffer(5)]],device uint* visits [[buffer(6)]],device const float* seed [[buffer(7)]],
    uint group [[threadgroup_position_in_grid]],uint simd [[simdgroup_index_in_threadgroup]],uint lane [[thread_index_in_simdgroup]]) {
    ulong column=ulong(group)*4+simd;if(column>=p.columns)return;
    ulong blocks=p.inner/128,stride=blocks+2,seedRows=min(ulong(8192),p.columns);
    if(column<seedRows) {
        if(lane==0){out[column]=seed[column];if(p.rows==2)out[p.oStride+column]=seed[seedRows+column];visits[column]=0;visits[p.columns+column]=p.rows==2?3:1;}
        return;
    }
    float sum0=0,sum1=0;bool alive0=true,alive1=p.rows==2;uint evaluated=0;
    float threshold0=stats[0],threshold1=p.rows==2?stats[stride]:0;
    float fullWeight=remaining[column*(blocks+1)];
    float scaled=float(2*p.inner+64)*5.960464477539063e-8f;
    float allowance=2*scaled/(1-scaled)+0.00001f;
    float error0=allowance*sqrt(fullWeight*stats[1]);
    float error1=p.rows==2?allowance*sqrt(fullWeight*stats[stride+1]):0;
    for(ulong block=0;block<blocks&&(alive0||alive1);++block) {
        ulong k=block*128+lane*4;
        ushort4 bits=*(device const ushort4*)(weight+column*p.inner+k);
        float4 w=as_type<float4>(uint4(bits)<<16),x=*(device const float4*)(a+k);
        float4 y=p.rows==2?*(device const float4*)(a+p.aStride+k):float4(0);
        for(uint j=0;j<4;++j){if(alive0)sum0+=x[j]*w[j];if(alive1)sum1+=y[j]*w[j];}
        ++evaluated;float partial0=simd_sum(sum0),partial1=p.rows==2?simd_sum(sum1):0;
        float residual=remaining[column*(blocks+1)+block+1];
        if(alive0&&partial0+sqrt(residual*stats[block+2])*1.001f+error0<threshold0)alive0=false;
        if(alive1&&partial1+sqrt(residual*stats[stride+block+2])*1.001f+error1<threshold1)alive1=false;
    }
    float result0=simd_sum(sum0),result1=p.rows==2?simd_sum(sum1):0;
    if(lane==0) {
        // A finite sentinel lets the unchanged native argmax reject actual
        // nonfinite values. These are not public complete vocabulary logits.
        out[column]=alive0?result0:-3.402823466e+38f;
        if(p.rows==2)out[p.oStride+column]=alive1?result1:-3.402823466e+38f;
        visits[column]=evaluated;visits[p.columns+column]=uint(alive0)|(uint(alive1)<<1);
    }
}
