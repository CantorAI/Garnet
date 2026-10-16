"""Reproduce source/layout preflight and exhaust all 24-bit words across uint3 load boundaries."""
from pathlib import Path
import hashlib,json,sys
import numpy as np
import make_sixbit_screen_index as factory
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);base=root/'src/backends/metal';tools=Path(__file__).parent
texts={'metal_context.h':factory.header((base/'metal_context.h').read_text()),'metal_context.mm':factory.context((base/'metal_context.mm').read_text()),'screened_head_index.h':factory.index((base/'screened_head_index.h').read_text()),'screened_head_native_impl.h':factory.native_impl((base/'screened_head_native_impl.h').read_text()),'nibble_screened_head_shader.h':factory.shader((base/'screened_head_shader.h').read_text()),'nibble_screen_index_gpu.mm':(tools/'sixbit_screen_index_gpu.mm').read_text()}
for name,text in texts.items():(out/name).write_bytes(text.encode())
cases=[]
for n,k in ((1,128),(3,128),(4,896),(5,1024),(17,37)):
 padded=(n+3)//4*4;v=np.zeros((padded,k),dtype=np.int8);v[:n]=(np.arange(n*k).reshape(n,k)%63-31).astype(np.int8);tags=(v.reshape(padded//4,4,k).transpose(0,2,1).astype(np.int32)&63).astype(np.uint32);words=tags[:,:,0]|(tags[:,:,1]<<6)|(tags[:,:,2]<<12)|(tags[:,:,3]<<18);packed=np.stack([(words>>(8*j))&255 for j in range(3)],axis=-1).astype(np.uint8).reshape(-1);w=packed.reshape(-1,3).astype(np.uint32);u=w[:,0]|(w[:,1]<<8)|(w[:,2]<<16);decoded=np.stack([(u>>(6*j))&63 for j in range(4)],axis=-1).astype(np.int16);decoded[decoded>=32]-=64;restored=decoded.reshape(padded//4,k,4).transpose(0,2,1).reshape(padded,k)
 assert np.array_equal(restored,v) and packed.size==padded*k*3//4;cases.append(dict(rows=n,inner=k,padded_rows=padded,index_bytes=int(packed.size)))
# Four adjacent 24-bit words span three 32-bit loads. Every possible word appears
# at every position, with rotated neighboring values to exercise both boundaries.
mask=np.uint32(0xffffff);checked=0
for base_value in range(0,1<<24,1<<16):
 v=np.arange(base_value,base_value+(1<<16),dtype=np.uint32);values=[v]+[((v<<j)|(v>>(24-j)))&mask for j in (5,9,13)]
 x=values[0]|(values[1]<<24);y=(values[1]>>8)|(values[2]<<16);z=(values[2]>>16)|(values[3]<<8)
 actual=[x&mask,((x>>24)|(y<<8))&mask,((y>>16)|(z<<16))&mask,(z>>8)&mask]
 assert all(np.array_equal(a,b) for a,b in zip(actual,values));checked+=4*len(v)
proof=dict(scope=__doc__,layout_cases=cases,all_24bit_words_checked_at_each_of_four_positions=True,word_comparisons=checked,generated_sha256={n:hashlib.sha256(t.encode()).hexdigest() for n,t in texts.items()},gpu_executed=False,full_goal_met=False)
(out/'preparation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
