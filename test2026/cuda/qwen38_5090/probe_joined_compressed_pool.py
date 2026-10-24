"""Joined gate/up pool checkpoint byte oracle, ownership and lifetime; no timing."""
import array,hashlib,json,os,struct,sys
from pathlib import Path
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT']);sys.path.insert(0,os.getcwd())
import garnet
from test2026.cuda.qwen38_5090.run_logged import source_identity
root,out=map(Path,sys.argv[1:3]);out.mkdir(exist_ok=True);path=out/'report.json'
if path.exists():raise FileExistsError('Preserve previous compressed pool attempt')
checkpoint=root/'models/Qwen3.8-27B-NVFP4'
index=json.loads((checkpoint/'model.safetensors.index.json').read_text())['weight_map']
joined='model.language_model.layers.0.mlp.gate_up_proj.weight'
names=[joined]
report={'status':'starting','source':source_identity(),'cases':[], 'scope':'One actual joined gate/up matrix: concatenated original packed/tiled bytes, readonly flags/rejection, resident accounting/cache reuse and retained allocation after close. No projection numerical, dual-engine, full model or timing claim.'}
io=garnet.cuda_tensor_io();held=[]
def save():path.write_text(json.dumps(report,indent=2))
def raw_tensor(t):
 values=garnet.tensor_to_cpu(t).tolist();a=array.array('i',values)
 if sys.byteorder!='little':a.byteswap()
 return a.tobytes()
def original_checkpoint_raw(name):
 with (checkpoint/index[name]).open('rb') as f:
  size=struct.unpack('<Q',f.read(8))[0];header=json.loads(f.read(size));entry=header[name];a,b=entry['data_offsets'];f.seek(8+size+a);raw=f.read(b-a)
 assert len(raw)==b-a
 return raw,entry
def checkpoint_raw(name):
 if name not in (joined,joined[:-7]+'.weight_scale'):return original_checkpoint_raw(name)
 suffix='.weight_scale' if name.endswith('.weight_scale') else '.weight'
 originals=[joined[:-len('gate_up_proj.weight')]+part+suffix for part in ('gate_proj','up_proj')]
 pieces=[original_checkpoint_raw(part) for part in originals]
 entry=dict(pieces[0][1]);entry['shape']=[pieces[0][1]['shape'][0]*2,pieces[0][1]['shape'][1]]
 return pieces[0][0]+pieces[1][0],entry
keys=['compressed_projection_weight_bytes','compressed_projection_scale_bytes','compressed_projection_pool_entries','compressed_nvfp4_entries','compressed_fp8_entries']
def counters():
 m=json.loads(io.memory_json());return {k:m[k] for k in keys}
try:
 io.open(str(checkpoint));report['initial']=counters();assert all(v==0 for v in report['initial'].values())
 for name in names:
  meta=json.loads(io.compressed_weight_info(name));assert meta['recipe']==('modelopt_nvfp4_w4a4' if '.mlp.' in name else 'modelopt_fp8_w8a8');assert not meta['resident_in_compressed_pool']
 assert counters()==report['initial'],'Metadata query allocated compressed GPU storage'
 for name in ['model.embed_tokens.weight','lm_head.weight','model.language_model.layers.0.input_layernorm.weight','model.language_model.layers.999.mlp.gate_proj.weight']:
  try:io.compressed_weight_info(name)
  except Exception:pass
  else:raise AssertionError('Invalid/noncompressed matrix admitted: '+name)
 assert counters()==report['initial']
 total_packed=total_scales=0
 for name in names:
  meta=json.loads(io.compressed_weight_info(name));packed=io.compressed_weight(name);scale=io.compressed_scales(name) if meta['scale_bytes'] else None
  expected,entry=checkpoint_raw(name);actual=raw_tensor(packed);assert actual==expected,'Packed bytes differ from checkpoint'
  item={'name':name,'metadata':meta,'packed_sha256':hashlib.sha256(actual).hexdigest(),'packed_bytes':len(actual)}
  total_packed+=len(actual);total_scales+=meta['scale_bytes']
  if scale is not None:
   original,_=checkpoint_raw(name[:-7]+'.weight_scale');rows,columns=meta['logical_shape'];blocks=columns//16;padded=((blocks+3)//4)*4;tiled=bytearray(meta['scale_bytes'])
   for row in range(rows):
    for col in range(blocks):tiled[(row//128)*128*padded+(col//4)*512+(row%32)*16+((row%128)//32)*4+col%4]=original[row*blocks+col]
   actual_scale=raw_tensor(scale);assert actual_scale==tiled,'Tiled scales differ from independent checkpoint repack';item['tiled_scale_sha256']=hashlib.sha256(actual_scale).hexdigest()
  else:
   try:io.compressed_scales(name)
   except Exception:pass
   else:raise AssertionError('FP8 block scales admitted')
  loaded=json.loads(io.compressed_weight_info(name));item['resident_metadata']=loaded;assert loaded['resident_packed_readonly'] and (scale is None or loaded['resident_scales_readonly'])
  try:io.copy_into(packed,packed)
  except Exception as error:item['writable_copy_rejection']=str(error)
  else:raise AssertionError('Readonly compressed tensor accepted by writable copy API')
  before=counters()
  for repeat in range(3):
   again=io.compressed_weight(name)
   if scale is not None:io.compressed_scales(name)
   assert counters()==before,'Repeat lookup changed resident accounting'
  item['accounting_before_repeats']=before;item['accounting_after_repeats']=counters()
  assert raw_tensor(again)==expected
  held.append((packed,scale,item['packed_sha256'],item.get('tiled_scale_sha256')));report['cases'].append(item);save()
 for member in json.loads(io.compressed_weight_info(joined))['source_weights']:
  try:io.compressed_weight(member)
  except Exception:pass
  else:raise AssertionError('Joined pool duplicated an original resident projection')
 final=counters();assert final['compressed_projection_weight_bytes']==total_packed and final['compressed_projection_scale_bytes']==total_scales
 assert final['compressed_projection_pool_entries']==1 and final['compressed_nvfp4_entries']==1 and final['compressed_fp8_entries']==0
 report['loaded_accounting']=final;io.close();report['closed_accounting']=counters();assert all(v==0 for v in report['closed_accounting'].values())
 for packed,scale,ph,sh in held:
  assert hashlib.sha256(raw_tensor(packed)).hexdigest()==ph
  if scale is not None:assert hashlib.sha256(raw_tensor(scale)).hexdigest()==sh
 report['retained_after_close_exact']=True
 try:io.compressed_weight(names[0])
 except Exception:pass
 else:raise AssertionError('Closed pool admitted lookup')
 report['status']='passed';print('JOINED_NVFP4_PACKED_TILED_BYTES_READONLY_CACHE_LIFETIME_PASS',flush=True)
except Exception as error:report['status']='failed';report['error']=repr(error);raise
finally:save();io.close()
