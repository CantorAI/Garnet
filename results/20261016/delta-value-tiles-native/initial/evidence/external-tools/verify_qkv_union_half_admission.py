"""Audit original-codec learned QKV provenance/counts; omitted checkpoint arithmetic is not rerun."""
from pathlib import Path
import hashlib,json,sys
base=Path(sys.argv[1]).resolve();root=base/'evidence';proof=json.loads((root/'manifest.json').read_text())
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
for name,info in proof['files'].items():
 p=(base/name).resolve();assert p.is_relative_to(base) and p.stat().st_size==info['bytes'] and digest(p)==info['sha256']
trial='qkv-union-half-admission-20261016';folder=root/'logs'/trial;assert set(proof['jobs'])=={trial} and proof['jobs'][trial]['exit_code']==0
assert json.loads((folder/'exit.json').read_text())==proof['jobs'][trial]
meta=json.loads((folder/'metadata.json').read_text());snap=folder/'source-snapshot'
for name,sha in meta['source_sha256'].items():assert digest(snap/Path(name).name)==sha
assert digest(snap/'exact_half_codec.h')=='e8800f4983bb2786a7e376ca7784a7817176a5e6dd2d46c7e353116f99b2a4c9'
original=(snap/'joined_union_half_admission_cpu.mm').read_text();assert original.count('side<2')==1 and original.count('pairs==76&&matrices==152')==1
generated=original.replace('side<2','side<3').replace('pairs==76&&matrices==152','pairs==58&&matrices==174').replace('pair','group')
p=folder/'qkv_union_half_admission_cpu.mm';assert p.read_bytes()==generated.encode() and digest(p)==meta['generated_sha256']
assert proof['omitted']['logs/'+trial+'/qkv_union_half_admission_cpu']['sha256']==meta['binary_sha256']
stages=json.loads((folder/'statuses.json').read_text());assert [v['name'] for v in stages]==['compile','audit'] and all(v['exit_code']==0 for v in stages)
assert 'Foundation' in stages[0]['command'] and not any(v in stages[0]['command'] for v in ('Metal','MetalPerformanceShaders'))
records=[json.loads(line) for line in (folder/'audit.log').read_text().splitlines()];assert len(records)==59 and records[-1]['groups']==58 and records[-1]['matrices']==174
prior={v['model']:v for v in json.loads((snap/'report.json').read_text())['models']};report=json.loads((folder/'report.json').read_text());models=[];restored=rejected=admitted=keys=0
for model,count in zip(report['models'],(28,24,6)):
 name=model['model'];header_path=folder/(name+'-header.json');header=json.loads(header_path.read_text());groups=model['groups'];assert len(groups)==count
 assert model['checkpoint_sha256']==prior[name]['checkpoint_sha256']==proof['omitted']['models/'+name+'/model.safetensors']['sha256']
 assert [v['layer'] for v in groups]==(list(range(count)) if count!=6 else list(range(3,24,4)))
 assert len(model['skipped_layers'])==(18 if count==6 else 0)
 for group in groups:
  assert len(group['sides'])==len(group['checks'])==3 and group['union_scale']==min(group['independent_scales'])
  cpp=next(v for v in records[:-1] if (v['model'],v['layer'])==(name,group['layer']));assert cpp['kind']=='group' and cpp['union_scale']==group['union_scale'] and cpp['admitted']==group['union_admitted']
  assert len(cpp['sides'])==3 and cpp['bitwise_roundtrip_values']==sum(v['encoded_roundtrip_values'] for v in cpp['sides']);restored+=cpp['bitwise_roundtrip_values'];admitted+=cpp['admitted']
  assert group['columns']==sum(v['rows'] for v in group['sides']) and group['logical_extra_union_bytes']==sum(v['original_bytes'] for v in group['sides'])
  for part,side,check,observed in zip(('q','k','v'),group['sides'],group['checks'],cpp['sides']):
   item=header[side['tensor']];assert side['part']==part and side['tensor'].endswith('.'+part+'_proj.weight') and item['dtype']=='BF16' and item['shape']==[side['rows'],group['inner']]
   assert side['data_offset']==8+header_path.stat().st_size+item['data_offsets'][0] and side['original_bytes']==side['rows']*group['inner']*2
   assert observed['tensor']==side['tensor'] and observed['raw_sha256']==side['raw_sha256']
   assert observed['encoded_roundtrip_values']+observed['rejected_values']==side['rows']*group['inner'] and check['unrepresentable_values']==observed['rejected_values'];rejected+=observed['rejected_values']
   assert check['admitted']==check['bitwise_roundtrip']==(observed['rejected_values']==0)
   assert side['retained_float_key']==(part=='k' and count!=6);keys+=side['retained_float_key']
  assert group['union_admitted']==all(v['admitted'] for v in group['checks'])
 models.append(dict(model=name,separate_groups=count,admitted_groups=sum(v['union_admitted'] for v in groups),existing_combined_layers=len(model['skipped_layers']),logical_extra_union_bytes=sum(v['logical_extra_union_bytes'] for v in groups if v['union_admitted'])))
assert records[-1]['admitted_groups']==admitted and records[-1]['bitwise_roundtrip_values']==restored and keys==52
result=dict(verified_files=len(proof['files']),groups=58,matrices=174,admitted_groups=admitted,captured_original_roundtrip_values=restored,rejected_values=rejected,retained_fp32_key_matrices=keys,models=models,scope='Captured CPU exact-value eligibility/provenance only; no learned GPU arithmetic, original envelope, graph integration or speed admission.',full_goal_met=False)
Path(sys.argv[2]).write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
