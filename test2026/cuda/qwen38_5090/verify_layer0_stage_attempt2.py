from pathlib import Path
import argparse,hashlib,json,struct,zipfile
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path);a=p.parse_args();out=Path(__file__).parent;archive=out/'layer0-real-stage-attempt2-evidence.zip'
with zipfile.ZipFile(archive) as z,zipfile.ZipFile(out/'layer0-real-stage-attempt1-evidence.zip') as failed,zipfile.ZipFile(out/'native-prefix12-internals-evidence.zip') as history:
 assert z.testzip() is None;e=json.loads(z.read('evidence-manifest.json'))
 for n,h in e['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
 for n,h in e['linked_archives'].items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h,n
 assert e['terminal']['event']=='finished' and e['terminal']['exit_code']==0
 base='results/layer0-real-stage-attempt2/';r=json.loads(z.read(base+'native/report.json'));assert r['status']=='witness_passed_needs_independent_array_replay' and r['library_sha256']==e['library_sha256']=='34cb221458064a4b7264586412be5b3c7acca888cd9e74681ac2a41d259413da'
 fix=json.loads(z.read('results/layer0-stage-probe-fix-install/receipt.json'));name='Garnet/test2026/cuda/qwen38_5090/probe_layer0_real_stages.py';assert hashlib.sha256(z.read(name)).hexdigest()==fix['probe_sha256'] and z.read(name)==z.read('probe-layer0-real-stages-warmed.py')
 assert z.read('results/layer0-stage-probe-fix-install/original-probe.py')==failed.read(name) and hashlib.sha256(failed.read(name)).hexdigest()==fix['old_probe_sha256'] and fix['failure_archive_sha256']==e['linked_archives']['layer0-real-stage-attempt1-evidence.zip']
 for n in ('src/backends/tensorrt/qwen35_lowering.h','src/backends/tensorrt/trt_builder.cpp'):assert z.read('Garnet/'+n)==failed.read('Garnet/'+n)
 old=json.loads(history.read('results/native-prefix12-internals/report.json'));assert r['input_ids']==old['input_prompt_ids'][:16] and len(r['input_ids'])==16 and r['config_sha256']==old['config_sha256']
 expected_stages=[('masked_input',81920),('input_norm',81920),('qkv',163840),('z',98304),('beta',768),('a',768),('gdn_gated_output',98304),('attention_output',81920),('residual',81920),('mlp_input_norm',81920),('output_hidden',81920)];assert r['stages']==[list(v) for v in expected_stages];assert len(r['files'])==81
 for item in r['files']:
  raw=z.read(base+'native/'+item['name']);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];assert np.isfinite(np.frombuffer(raw,dtype='<f4')).all()
 def data(b,repeat,label,row=0):return z.read(base+'native/b'+str(b)+'-r'+str(repeat)+'-'+label+'-row'+str(row)+'.f32')
 for b in (1,2):
  for row in range(b):
   actual=data(b,0,'ordinary',row);assert len(actual)==909312*4 and actual==data(b,1,'ordinary',row)
   assert actual[:81920*4]==history.read('results/native-prefix12-internals/prefix-1-batch-'+str(b)+'-repeat-0-chunk-0-hidden-row-'+str(row)+'.f32')
   assert actual[81920*4:]==history.read('results/native-prefix12-internals/prefix-1-batch-'+str(b)+'-repeat-0-chunk-0-layer-0-state-row-'+str(row)+'.f32')
   for repeat in (0,1):
    assert data(b,repeat,'primary',row)==actual
    assert data(b,repeat,'masked_input',row)==data(b,-1,'prompt_input',row)
    assert data(b,repeat,'output_hidden',row)==actual[:81920*4]
    for label,width in expected_stages:
     assert len(data(b,repeat,label,row))==width*4 and data(b,repeat,label,row)==data(b,0,label,row)==data(b,repeat,label,0)
  assert data(b,0,'ordinary',0)==data(b,0,'ordinary',b-1)
 for record in r['native_projection_counts']:
  b=record['batch'];st=record['stats'];assert st['native_nvfp4_projection_weights']==st['native_fp8_projection_weights']==3 and st['native_int4_g128_projection_weights']==0
  assert st['device_output_bytes']==b*(909312+sum(w for _,w in expected_stages))*4 and st['graph_capture_failures']==0 and st['graph_captures']==1 and st['graph_replays']>=1
 assert {(v['flag'],v['batch'],v['time']) for v in r['invalid_profiles']}=={(2,1,16),(-1,1,16),(1,4,16),(1,1,4)} and all(v['rejected'] for v in r['invalid_profiles'])
 assert len(r['warmup_boundaries'])==6
 for b in (1,2):
  boundary=[v for v in r['warmup_boundaries'] if v['batch']==b];assert [v['index'] for v in boundary]==[0,1,2] and all(v['bytes']==b*4 for v in boundary)
 embedding={'checked_against_checkpoint':False}
 if a.checkpoint:
  checkpoint=a.checkpoint;assert hashlib.sha256((checkpoint/'config.json').read_bytes()).hexdigest()==r['config_sha256'];index=json.loads((checkpoint/'model.safetensors.index.json').read_text());key='model.language_model.embed_tokens.weight';shard=checkpoint/index['weight_map'][key]
  with shard.open('rb') as f:
   length=struct.unpack('<Q',f.read(8))[0];header=f.read(length);info=json.loads(header)[key];assert info['dtype']=='BF16' and info['shape'][1]==5120;rows=[];hashes=[]
   for token in r['input_ids']:
    assert 0<=token<info['shape'][0];f.seek(8+length+info['data_offsets'][0]+token*5120*2);raw=f.read(5120*2);assert len(raw)==5120*2;hashes.append(hashlib.sha256(raw).hexdigest());rows.append((np.frombuffer(raw,dtype='<u2').astype('<u4')<<16).tobytes())
  full=b''.join(rows)
  for b in (1,2):
   for row in range(b):assert data(b,-1,'prompt_input',row)==full
  embedding={'checked_against_checkpoint':True,'tensor':key,'shard':shard.name,'header_sha256':hashlib.sha256(header).hexdigest(),'raw_bf16_row_sha256':hashes,'input_f32_sha256':hashlib.sha256(full).hexdigest()}
 comparisons=[]
 for label,width in [('ordinary',909312),('primary',909312)]+expected_stages:
  first=np.frombuffer(data(1,0,label),dtype='<f4');second=np.frombuffer(data(2,0,label),dtype='<f4');assert first.size==second.size==width;changed=np.flatnonzero(first.view('<u4')!=second.view('<u4'));delta=first.astype('float64')-second.astype('float64')
  comparisons.append({'stage':label,'words':width,'changed_words':int(changed.size),'max_abs':float(np.max(np.abs(delta))),'rms':float(np.sqrt(np.mean(delta*delta))),'first_changed_indices':[int(i) for i in changed[:16]]})
 first_changed=next(v['stage'] for v in comparisons[2:] if v['changed_words'])
 for folder in ('probe-ledger',):
  events=[json.loads(l) for l in z.read(base+folder+'/commands.jsonl').decode().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
 resource=json.loads(z.read(base+'resources/resource-monitor.json'));samples=[json.loads(l) for l in z.read(base+'resources/resource-vram.jsonl').decode().splitlines()];assert resource['child_exit_code']==0 and resource['sampled_memory']['samples']==len(samples) and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==max(v['used_bytes'] for v in samples)
 result={'status':'actual_layer0_stage_witness_verified','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files_verified':len(e['files']),'library_sha256':e['library_sha256'],'primary_equals_ordinary_all_rows_repeats':True,'historical_hidden_state_all_exact':True,'embedding':embedding,'comparisons':comparisons,'first_observable_changed_stage':first_changed,'scope':'Actual first16 prompt tokens, prefix1 B1/B2, two retained repeats/duplicate rows; complete primary and historical states preserved. First observable difference locates the FP8 output-projection path when attention_output changes; cast/quantize/matmul/restore/select are not yet separated. Not a unique algorithm cause, full64-layer batch admission, model speed or2x proof.'}
(out/'layer0-real-stage-attempt2-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
