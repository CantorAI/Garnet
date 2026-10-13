"""Generate exact-value FP32 SIMD-group prefill tile variants, plus control."""
from pathlib import Path
import sys

TILES=((16,32,64),(16,32,128),(16,64,64),(32,32,64),(32,64,32))

def name(tile):return 'gemm_'+str(tile[0])+'_'+str(tile[1])+'_'+str(tile[2])

def kernel(tile):
    rows,columns,inner=tile;threads=columns//8*32;groups=rows//8
    assert (rows*inner+inner*columns+rows*columns)*4<=32768
    lines=[f'kernel void {name(tile)}(device const float* a [[buffer(0)]],device const ushort* weight [[buffer(1)]],',
        'device float* out [[buffer(2)]],constant BF16Params& p [[buffer(3)]],',
        'uint2 group [[threadgroup_position_in_grid]],uint tid [[thread_index_in_threadgroup]],uint simd [[simdgroup_index_in_threadgroup]]) {',
        f'threadgroup float left[{rows*inner}],right[{inner*columns}],result[{rows*columns}];',
        f'ulong rowBase=ulong(group.y)*{rows},colBase=ulong(group.x)*{columns};']
    for i in range(groups):lines.append(f'simdgroup_float8x8 accum{i}=make_filled_simdgroup_matrix<float,8>(0.0f);')
    lines.extend([f'for(ulong base=0;base<p.inner;base+={inner}) {{',
        f'for(uint i=tid;i<{rows*inner};i+={threads}){{ulong row=rowBase+i/{inner},k=base+i%{inner};left[i]=(row<p.rows&&k<p.inner)?a[row*p.aStride+k]:0.0f;}}',
        f'for(uint i=tid;i<{inner*columns};i+={threads}){{ulong k=base+i/{columns},column=colBase+i%{columns};right[i]=(k<p.inner&&column<p.columns)?as_type<float>(uint(weight[column*p.inner+k])<<16):0.0f;}}',
        'threadgroup_barrier(mem_flags::mem_threadgroup);',f'for(uint offset=0;offset<{inner};offset+=8) {{',
        'simdgroup_float8x8 r;',f'simdgroup_load(r,right+offset*{columns}+simd*8,{columns});'])
    for i in range(groups):
        lines.extend([f'simdgroup_float8x8 l{i};',f'simdgroup_load(l{i},left+{i*8*inner}+offset,{inner});',f'simdgroup_multiply_accumulate(accum{i},l{i},r,accum{i});'])
    lines.extend(['}','threadgroup_barrier(mem_flags::mem_threadgroup);','}'])
    for i in range(groups):lines.append(f'simdgroup_store(accum{i},result+{i*8*columns}+simd*8,{columns});')
    lines.extend(['threadgroup_barrier(mem_flags::mem_threadgroup);',
        f'for(uint i=tid;i<{rows*columns};i+={threads}){{ulong row=rowBase+i/{columns},column=colBase+i%{columns};if(row<p.rows&&column<p.columns)out[row*p.oStride+column]=result[i];}}','}'])
    return '\n'.join(lines)+'\n'

def main():
    text=Path(sys.argv[1]).read_text();begin=text.index('struct BF16Params');end=text.index('struct ArgmaxParams')
    Path(sys.argv[2]).write_text('#include <metal_stdlib>\nusing namespace metal;\n'+text[begin:end]+''.join(kernel(tile) for tile in TILES))

if __name__=='__main__':main()
