from pathlib import Path
import hashlib,io,json,statistics,subprocess,sys,zipfile
out=Path(__file__).parent;repo=Path('D:/CantorAI/work/qwen38-5090');archive=out/'native-controls-abba-evidence.zip'
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None;e=json.loads(z.read('evidence-manifest.json'))
 for n,h in e['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
 assert e['terminal']['event']=='finished' and e['terminal']['exit_code']==0
 for n,h in e['linked_archives'].items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h,n
 base='results/native-controls-abba/';plan=json.loads(z.read(base+'plan.json'));completed=json.loads(z.read(base+'completed.json'))
 assert completed['status']=='four_legs_finished_needs_independent_reaccounting'
 assert [(v['leg'],v['resident_controls']) for v in plan['order']]==[('a1',False),('b1',True),('b2',True),('a2',False)]
 assert [v['leg'] for v in completed['legs']]==['a1','b1','b2','a2']
 assert z.read('controls-abba-experiment-plan.json')==z.read(base+'plan.json')==(repo/'test2026/cuda/qwen38_5090/resident_controls_abba_plan.json').read_bytes()
 assert z.read('run-native-controls-abba.py')==(repo/'test2026/cuda/qwen38_5090/run_native_controls_abba.py').read_bytes()
 recipe=json.loads((repo/'test2026/cuda/qwen38_5090/resident_controls_abba_recipe.json').read_text());assert hashlib.sha256(z.read('controls-abba-methods.zip')).hexdigest()==recipe['package_sha256']
 with zipfile.ZipFile(io.BytesIO(z.read('controls-abba-methods.zip'))) as methods:
  hashes=json.loads(methods.read('methods-sha256.json'))
  for n,h in hashes.items():assert hashlib.sha256(methods.read(n)).hexdigest()==h,n
  assert methods.read('controls-abba-experiment-plan.json')==z.read(base+'plan.json')
  template=(out/'verify-native-resident-controls-http.py').read_text(encoding='utf8');legs={};timestamps=[];telemetry={}
  for spec,record in zip(plan['order'],completed['legs']):
   leg=spec['leg'];prefix=spec['prefix'];assert record['workflow_exit_code']==0
   name=prefix+'-http-evidence.zip';assert e['linked_archives'][name]==record['archive_sha256']
   with zipfile.ZipFile(out/name) as raw:
    manifest=json.loads(raw.read('evidence-manifest.json'));assert manifest['library_sha256']==e['library_sha256']
    for n,h in plan['source_sha256'].items():assert hashlib.sha256(raw.read(n)).hexdigest()==h,n
    for name in hashes:
     if prefix in name:assert methods.read(name)==raw.read(name),name
    event=manifest['terminal'][prefix+'-http-controller-ledger'];timestamps.append(event['utc'])
   code=template.replace('native-resident-controls',prefix).replace('shared-resident-controls-http-diagnostic','shared-controls-abba-'+leg+'-http-diagnostic')
   if not spec['resident_controls']:code=code.replace("worker['resident_request_controls'] is True","worker['resident_request_controls'] is False")
   script=out/('verify-'+prefix+'-http.py');script.write_text(code,encoding='utf8')
   process=subprocess.run([sys.executable,str(script)],capture_output=True,text=True);(out/(prefix+'-http-verification.log')).write_text(process.stdout+process.stderr,encoding='utf8');assert process.returncode==0,leg+' verification failed; inspect retained log'
   verified=json.loads((out/(prefix+'-http-verified.json')).read_text());assert verified['status']=='resident_controls_http_evidence_verified';legs[leg]=verified
   for label in ('before','after'):
    t=json.loads(z.read(base+leg+'-'+label+'-gpu.json'));assert t['exit']==0;telemetry[leg+'-'+label]=t
 assert timestamps==sorted(timestamps) and len(set(timestamps))==4
 workload=legs['a1']['run']
 for leg,v in legs.items():
  assert v['run']['transport']==workload['transport']
  for key,cohort in v['run']['cohorts'].items():assert cohort['requests']==workload['cohorts'][key]['requests'],(leg,key)
 table={}
 for case in ('text-c1','image-c1'):
  table[case]={}
  for metric in ('decode_median_tokens_per_second','ttft_median_ms','output_tokens_per_second'):
   values={leg:v['run']['cohorts'][case]['metrics'][metric] for leg,v in legs.items()};a=[values[k] for k in ('a1','a2')];b=[values[k] for k in ('b1','b2')]
   table[case][metric]={'per_leg':values,'off_mean_of_two_cohort_summaries':statistics.mean(a),'on_mean_of_two_cohort_summaries':statistics.mean(b),'on_off_ratio_of_means':statistics.mean(b)/statistics.mean(a),'off_range':[min(a),max(a)],'on_range':[min(b),max(b)],'mirrored_pair_ratios':[values['b1']/values['a1'],values['b2']/values['a2']]}
 result={'status':'four_leg_controls_abba_evidence_verified','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'parent_files_verified':len(e['files']),'legs':{k:{'archive_sha256':v['archive_sha256'],'files_verified':v['files_verified'],'strict_gates':v['strict_gates'],'measured_ids_exact':v['measured_ids_exact']} for k,v in legs.items()},'same_binary_source_and_workload_verified':True,'table':table,'telemetry':telemetry,'two_x_established':False,'scope':'Prespecified four fresh OFF/ON/ON/OFF same-binary B1 pilots; each20target/MTP+20oldIDs+2SSE+16measuredSSE+NVML. Two cohort summaries per mode, not enough for broad statistical claims. Report raw ranges/pair ratios; telemetry only before/after, not continuous timed clock/temperature proof. No default promotion, tuned-reference/broad-quality/full-context/trueCUDAconcurrency or2x admission.'}
(out/'native-controls-abba-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
