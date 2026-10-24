from pathlib import Path
import collections,hashlib,json,zipfile
out=Path(__file__).parent;a=out/'warm-mtp-resident-controls-attempt1-evidence.zip'
with zipfile.ZipFile(a) as z,zipfile.ZipFile(out/'warm-mtp-trace-unprofiled-control-evidence.zip') as control,zipfile.ZipFile(out/'native-mtp3-decode-only-off-http-evidence.zip') as off:
 assert z.testzip() is None;e=json.loads(z.read('evidence-manifest.json'))
 for n,h in e['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
 for n,h in e['linked_archives'].items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h,n
 assert e['terminal']['event']=='finished' and e['terminal']['exit_code']==0
 def text(archive,name):return archive.read(name).decode('utf8').replace('\r\n','\n')
 expected=text(control,'probe-warm-mtp-trace-startup.py').replace('decoder.enable_cuda_graph(True);io=decoder.io','decoder.resident_controls=True\n decoder.enable_cuda_graph(True);io=decoder.io').replace("report['status']='warming';save()","report['resident_controls']=bool(decoder.resident_controls)\n report['status']='warming';save()")
 assert expected==text(z,'probe-warm-mtp-resident-controls.py')
 for n in e['files']:
  if n.startswith('Garnet/') and n in control.namelist():assert z.read(n)==control.read(n),n
 base='results/warm-mtp-resident-controls-attempt1/'
 workflow=json.loads(z.read(base+'workflow.json'));assert workflow['status']=='producer_correctness_passed_needs_independent_verification'
 report=json.loads(z.read(base+'native/report.json'));assert report['status']=='passed' and report['resident_controls'] is True and len(report['cases'])==14
 assert report['source']['commit']=='754cfa42473dd106a584c5e9584a54e8d994169b'
 assert report['library_sha256']==e['library_sha256']=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
 counts=collections.Counter()
 for row in report['cases']:
  key=row['key'];raw=off.read('results/shared-decode-only-off-http-diagnostic/'+key+'-target-response.json');request=off.read('results/shared-decode-only-off-http-diagnostic/'+key+'-target-request.json');teacher=json.loads(raw)['choices'][0]
  assert row['generated_ids']==teacher['response_token_ids'] and row['teacher_sha256']==hashlib.sha256(raw).hexdigest() and row['request_sha256']==hashlib.sha256(request).hexdigest()
  assert row['metrics']['target_head_reuses']==0 and row['reuse_verified_head'] is False;counts[(row['phase'],key)]+=1
  if key in ('red','green','blue','bottom-blue'):assert row['text'].strip().lower().rstrip('.')==('blue' if key=='bottom-blue' else key)
 expected={('warmup',k):2 for k in ['s3-text-c1-r0','s3-image-c1-r0','red','green','blue','bottom-blue']};expected.update({('capture',k):1 for k in ['s3-text-c1-r0','s3-image-c1-r0']});assert counts==expected
 transport=json.loads(z.read(base+'transport.json'));assert transport['all_passed'] and len(transport['records'])==16
 profiles={(b,t) for b in (1,2,4,8) for t in (4,16,64,128)};assert {(r['batch'],r['chunk']) for r in transport['records']}==profiles
 for row in transport['records']:assert row['trials']==3 and row['controls_and_embeddings_byte_exact'] and row['cpu_argument_lifetime_and_padding'] and row['invalid_inputs_preserve_outputs']
 resources={}
 for stage in ('transport','model'):
  events=[json.loads(l) for l in z.read(base+stage+'-ledger/commands.jsonl').decode().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
  assert not any('nsys' in v.lower() for v in events[0]['argv'])
  resource=json.loads(z.read(base+stage+'-resources/resource-monitor.json'));samples=[json.loads(l) for l in z.read(base+stage+'-resources/resource-vram.jsonl').decode().splitlines()]
  assert resource['child_exit_code']==0 and resource['sampled_memory']['samples']==len(samples) and resource['sampled_memory']['sampled_peak_adapter_used_bytes']==max(v['used_bytes'] for v in samples);resources[stage]=resource
 result={'status':'resident_controls_correctness_evidence_verified','archive_sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'files_verified':len(e['files']),'exact_model_cases':14,'producer_transport_profiles':16,'producer_transport_trials':48,'resources':resources,'scope':'Independent archive/source/runtime/14 full model output comparisons and producer transport assertions/terminal ledgers verified. Transport tensors themselves are not archived, so bitwise transport proof remains producer assertions. No HTTP, uninstrumented speed, true CUDA batching, broad-quality or2x admission.'}
(out/'warm-mtp-resident-controls-attempt1-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
