"""Original FP32 matrix kernel at decode rows, plus an eight-row internal tile."""
from pathlib import Path
import sys
native=Path(sys.argv[1]).read_text()
def function(name):
 start=native.index('kernel void '+name+'(');brace=native.index('{',start);depth=1;end=brace+1
 while depth:
  depth+=(native[end]=='{')-(native[end]=='}');end+=1
 return native[start:end]+'\n'
gemm=function('bf16_gemm');eight=gemm.replace('bf16_gemm(','bf16_decode_matrix8(')
eight=eight.replace('left[16*32]','left[8*32]').replace('result[16*32]','result[8*32]').replace('group.y)*16','group.y)*8').replace('i<16*32','i<8*32')
eight=eight.replace('    simdgroup_float8x8 accum1=make_filled_simdgroup_matrix<float,8>(0.0f);\n','')
eight=eight.replace('simdgroup_float8x8 l0,l1,r;','simdgroup_float8x8 l0,r;').replace('simdgroup_load(l1,left+8*32+offset,32);','').replace('simdgroup_multiply_accumulate(accum1,l1,r,accum1);','').replace('simdgroup_store(accum1,result+8*32+simd*8,32);','')
assert 'accum1' not in eight and 'l1' not in eight
shader='#include <metal_stdlib>\nusing namespace metal;\nstruct BF16Params{ulong rows,inner,columns,aStride,oStride;};\n'+function('bf16_decode')+function('bf16_decode_small')+gemm+eight
Path(sys.argv[2]).write_bytes(shader.encode())
generator=Path(__file__).with_name('make_decode_subsimd_probe.py').read_text()
original_args=sys.argv;sys.argv=[sys.argv[0],sys.argv[3]]
exec(compile(generator,str(Path(__file__).with_name('make_decode_subsimd_probe.py')),'exec'))
sys.argv=original_args;path=Path(sys.argv[3]);text=path.read_text()
start=text.index('        std::vector<Tile> tiles=');end=text.index('        for(auto& tile:',start)
text=text[:start]+'''        std::vector<Tile> tiles={{"bf16_decode",1,16,4,128,nil},{"bf16_decode_matrix8",8,32,32,128,nil},{"bf16_gemm",16,32,32,128,nil}};
'''+text[end:]
needle='            auto weight=[device newBufferWithBytes:bytes.bytes length:bytes.length options:MTLResourceStorageModeShared];'
assert text.count(needle)==1
text=text.replace(needle,needle+'''
            tiles[0].name=columns<=256?"bf16_decode_small":"bf16_decode";
            tiles[0].columns=columns<=256?4:16;tiles[0].threads=columns<=256?64:128;
            tiles[0].pipeline=pipeline(device,library,tiles[0].name);
''')
path.write_bytes(text.encode())
