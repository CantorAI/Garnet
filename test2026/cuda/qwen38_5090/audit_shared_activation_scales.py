"""Read only pinned calibration scalars; no Torch, CUDA, model changes or benchmark."""
from pathlib import Path
import hashlib,json,math,struct,sys
root=Path(sys.argv[1]);out=Path(sys.argv[2]);checkpoint=root/'models/Qwen3.8-27B-NVFP4'
if out.exists():raise FileExistsError('Preserve earlier scalar audit')
index_raw=(checkpoint/'model.safetensors.index.json').read_bytes();index=json.loads(index_raw)['weight_map'];headers={};scalars={}
def read_scalar(name):
 if name in scalars:return scalars[name]
 shard=(checkpoint/index[name]).resolve();assert shard.is_relative_to(checkpoint.resolve())
 if shard not in headers:
  with shard.open('rb') as f:
   length=struct.unpack('<Q',f.read(8))[0];assert 0<length<=64<<20
   raw=f.read(length);assert len(raw)==length
  headers[shard]=(8+length,json.loads(raw),hashlib.sha256(raw).hexdigest())
 start,header,header_sha=headers[shard];entry=header[name]
 assert math.prod(entry['shape'])==1
 left,right=entry['data_offsets'];assert 0<=left<right and start+right<=shard.stat().st_size
 with shard.open('rb') as f:f.seek(start+left);raw=f.read(right-left)
 dtype=entry['dtype']
 if dtype=='F32':value=struct.unpack('<f',raw)[0]
 elif dtype=='F16':value=struct.unpack('<e',raw)[0]
 elif dtype=='BF16':value=struct.unpack('<f',b'\0\0'+raw)[0]
 else:raise ValueError('Unqualified scalar dtype: '+dtype)
 assert math.isfinite(value) and value>0
 row={'name':name,'dtype':dtype,'raw_hex':raw.hex(),'value':value,'shard':index[name],'data_offsets':[left,right],'header_sha256':header_sha}
 scalars[name]=row;return row
config=json.loads((checkpoint/'config.json').read_bytes())['text_config'];groups=[]
for i,kind in enumerate(config['layer_types']):
 prefix='model.language_model.layers.'+str(i)
 candidates=[('mlp_gate_up',[prefix+'.mlp.'+name for name in ('gate_proj','up_proj')])]
 if kind=='linear_attention':candidates.append(('gdn_qkv_z',[prefix+'.linear_attn.'+name for name in ('in_proj_qkv','in_proj_z')]))
 else:candidates.append(('attention_qkv',[prefix+'.self_attn.'+name for name in ('q_proj','k_proj','v_proj')]))
 for family,names in candidates:
  inputs=[read_scalar(name+'.input_scale') for name in names]
  weights=[read_scalar(name+('.weight_scale_2' if family=='mlp_gate_up' else '.weight_scale')) for name in names]
  groups.append({'layer':i,'family':family,'projections':names,'input_scale_names':[row['name'] for row in inputs],'weight_scale_names':[row['name'] for row in weights],
   'input_scales_bit_identical':len({(row['dtype'],row['raw_hex']) for row in inputs})==1,
   'weight_scales_bit_identical':len({(row['dtype'],row['raw_hex']) for row in weights})==1})
summary={family:{'groups':len(rows),'identical_input_scale_groups':sum(row['input_scales_bit_identical'] for row in rows),'identical_weight_scale_groups':sum(row['weight_scales_bit_identical'] for row in rows)} for family in ('mlp_gate_up','gdn_qkv_z','attention_qkv') for rows in ([row for row in groups if row['family']==family],)}
report={'status':'passed','index_sha256':hashlib.sha256(index_raw).hexdigest(),'summary':summary,'groups':groups,'scalars':scalars,
 'scope':'Read-only calibration eligibility audit. Equal input scalars permit investigating shared quantization but do not establish common activation tensors, supported fused kernels, exact arithmetic, correctness or speedup. Weight/global output scales must remain correct.'}
out.write_text(json.dumps(report,indent=2));print(json.dumps(summary))
