"""Reproduce the Windows-only arbitrary-bit tile addressing and source preflight."""
from pathlib import Path
import ast,hashlib,json,sys
import numpy as np
import make_block_packed_prefill as factory
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();out.mkdir(parents=True,exist_ok=False);tools=Path(__file__).parent;base=root/'src/backends/metal'
texts={'metal_context.h':factory.header((base/'metal_context.h').read_text()),'metal_context.mm':factory.context((base/'metal_context.mm').read_text()),'block_packed_prefill_shader.h':factory.shader(),'block_packed_prefill_gpu.mm':factory.harness((tools/'packed_body_prefill_gpu.mm').read_text())}
for name,text in texts.items():(out/name).write_bytes(text.encode())
cases=0;words=0
for n in (1,19,64,65,257):
 for k in (1,31,32,37,129):
  raw=np.random.default_rng(n*1000+k).integers(0,65536,size=(n,k),dtype=np.uint16);nt=(n+63)//64;kt=(k+31)//32;packet=np.zeros(nt*kt*2048,dtype=np.uint16)
  ni,ki=np.indices((n,k));address=((ni//64)*kt+ki//32)*2048+(ki%32)*64+ni%64
  assert len(np.unique(address))==n*k;packet[address]=raw
  padded=np.zeros((nt*64,kt*32),dtype=np.uint16);padded[:n,:k]=raw
  ni,ki=np.indices(padded.shape);all_address=((ni//64)*kt+ki//32)*2048+(ki%32)*64+ni%64
  assert len(np.unique(all_address))==packet.size and np.array_equal(packet[all_address],padded)
  cases+=1;words+=padded.size
for name in ('make_block_packed_prefill.py','run_block_packed_prefill.py'):ast.parse((tools/name).read_text())
proof=dict(cpu_layout_cases=cases,cpu_padded_words_checked=words,arbitrary_bf16_payloads=True,gpu_executed=False,generated_sha256={n:hashlib.sha256(t.encode()).hexdigest() for n,t in texts.items()},scope='Windows source generation and bijective N64/K32 address/bit/padding proof only; no numerical or GPU speed claim',full_goal_met=False)
(out/'preparation.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps(proof))
