"""Source and signed-nibble layout preflight only, no GPU speed or final-score claim."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import make_nibble_screen_index as factory
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);base=root/'src/backends/metal';tools=Path(__file__).parent
texts={'metal_context.h':factory.header((base/'metal_context.h').read_text()),'metal_context.mm':factory.context((base/'metal_context.mm').read_text()),'screened_head_index.h':factory.index((base/'screened_head_index.h').read_text()),'screened_head_native_impl.h':factory.native_impl((base/'screened_head_native_impl.h').read_text()),'nibble_screened_head_shader.h':factory.shader((base/'screened_head_shader.h').read_text()),'nibble_screen_index_gpu.mm':(tools/'nibble_screen_index_gpu.mm').read_text()}
for name,text in texts.items():(out/name).write_bytes(text.encode())
values=np.arange(256,dtype=np.uint8);lo=(values&15).astype(np.int16);hi=(values>>4).astype(np.int16);lo[lo>=8]-=16;hi[hi>=8]-=16;assert np.array_equal(((lo&15)|((hi&15)<<4)).astype(np.uint8),values)
cases=[]
for rows,inner in ((1,128),(3,128),(4,896),(5,1024),(17,37)):
 raw=(np.arange(rows*inner,dtype=np.int32).reshape(rows,inner)%15-7).astype(np.int8);padded=(rows+3)//4*4;original=np.zeros((padded,inner),dtype=np.int8);original[:rows]=raw
 interleaved=original.reshape(padded//4,4,inner).transpose(0,2,1).reshape(-1);tags=(interleaved.astype(np.int16)&15).astype(np.uint8);packed=tags[::2]|(tags[1::2]<<4)
 decoded=np.empty(len(tags),dtype=np.int16);decoded[::2]=packed&15;decoded[1::2]=packed>>4;decoded[decoded>=8]-=16;restored=decoded.reshape(padded//4,inner,4).transpose(0,2,1).reshape(padded,inner)
 assert np.array_equal(restored,original) and len(packed)*2==padded*inner
 cases.append(dict(rows=rows,inner=inner,padded_rows=padded,index_bytes=len(packed),padded_words=padded*inner))
proof=dict(scope=__doc__,signed_byte_patterns=256,layout_cases=cases,generated_sha256={n:hashlib.sha256(t.encode()).hexdigest() for n,t in texts.items()},gpu_executed=False,full_goal_met=False)
(out/'preparation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
