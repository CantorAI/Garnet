from pathlib import Path
import collections,hashlib,json,sqlite3,zipfile
out=Path(__file__).parent;archive=out/'warm-mtp-cuda-trace-2026-attempt1-evidence.zip'
with zipfile.ZipFile(archive) as z,zipfile.ZipFile(out/'native-mtp3-decode-only-off-http-evidence.zip') as off:
 assert z.testzip() is None;e=json.loads(z.read('evidence-manifest.json'))
 for n,h in e['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
 for n,h in e['linked_archives'].items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h,n
 assert e['terminal']['event']=='finished' and e['terminal']['exit_code']==0
 base='results/warm-mtp-cuda-trace-2026-attempt1/';workflow=json.loads(z.read(base+'workflow.json'));assert workflow['status']=='native_exact_marked_trace_unverified' and workflow['stats_exit_code']==0
 report=json.loads(z.read(base+'native/report.json'));assert report['status']=='passed' and len(report['cases'])==14
 assert report['library_sha256']==e['library_sha256']=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
 assert report['source']['commit']=='754cfa42473dd106a584c5e9584a54e8d994169b'
 counts=collections.Counter()
 for row in report['cases']:
  key=row['key'];teacher=json.loads(off.read('results/shared-decode-only-off-http-diagnostic/'+key+'-target-response.json'))['choices'][0]
  request=off.read('results/shared-decode-only-off-http-diagnostic/'+key+'-target-request.json');response=off.read('results/shared-decode-only-off-http-diagnostic/'+key+'-target-response.json')
  assert row['generated_ids']==teacher['response_token_ids'] and row['request_sha256']==hashlib.sha256(request).hexdigest() and row['teacher_sha256']==hashlib.sha256(response).hexdigest()
  assert row['reuse_verified_head'] is False and row['metrics']['target_head_reuses']==0
  counts[(row['phase'],key)]+=1
 expected={('warmup',k):2 for k in ['s3-text-c1-r0','s3-image-c1-r0','red','green','blue','bottom-blue']};expected.update({('capture',k):1 for k in ['s3-text-c1-r0','s3-image-c1-r0']});assert counts==expected
 for folder in ('profile-ledger','stats-ledger'):
  events=[json.loads(l) for l in z.read(base+folder+'/commands.jsonl').decode().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
 launcher=z.read('test-warm-mtp-trace-2026.bat').decode();assert 'GARNET_NVFP4_DOWN_CANDIDATE=0' in launcher
 path=out.parent/'work/warm-mtp-cuda-trace-2026-attempt1.sqlite';path.write_bytes(z.read(base+'startup-trace.sqlite'))
 c=sqlite3.connect('file:'+path.as_posix()+'?mode=ro',uri=True);c.row_factory=sqlite3.Row
 kernels=[dict(r) for r in c.execute('select k.*,s.value as name from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on k.demangledName=s.id order by k.start')];assert len(kernels)>5 and all(k['end']>=k['start'] for k in kernels)
 api=[dict(r) for r in c.execute('select r.*,s.value as name from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on r.nameId=s.id')];assert api
 def aggregate(rows):
  groups={}
  for row in rows:
   item=groups.setdefault(row['name'],{'name':row['name'],'count':0,'sum_duration_ns':0,'max_duration_ns':0});dt=row['end']-row['start'];item['count']+=1;item['sum_duration_ns']+=dt;item['max_duration_ns']=max(item['max_duration_ns'],dt)
  return sorted(groups.values(),key=lambda r:r['sum_duration_ns'],reverse=True)
 def union_ns(rows):
  total=0;left=right=None
  for row in sorted(rows,key=lambda v:v['start']):
   a,b=row['start'],row['end']
   if right is None:left,right=a,b
   elif a<=right:right=max(right,b)
   else:total+=right-left;left,right=a,b
  return total+(right-left if right is not None else 0)
 tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
 copies=[dict(r) for r in c.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY')] if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables else []
 markers=report['markers'];starts=[r for r in copies if r['copyKind']==1 and r['bytes']==markers['start_copy_bytes']];ends=[r for r in copies if r['copyKind']==1 and r['bytes']==markers['end_copy_bytes']]
 assert len(starts)==len(ends)==1,'Unique HtoD markers required; never infer a window from timing guesses'
 start,end=starts[0],ends[0];left,right=start['end'],end['start'];assert left<right and start['globalPid']==end['globalPid'] and start['deviceId']==end['deviceId']
 all_kernel_records=len(kernels);outside=sum(k['end']<=left or k['start']>=right for k in kernels)
 crossing=[k for k in kernels if k['start']<right and k['end']>left and not (k['start']>=left and k['end']<=right)];assert not crossing,'Kernel crosses diagnostic marker; scope requires investigation'
 kernels=[k for k in kernels if k['start']>=left and k['end']<=right and k['globalPid']==start['globalPid'] and k['deviceId']==start['deviceId']];assert len(kernels)>5
 api=[r for r in api if r['start']>=left and r['end']<=right];assert api
 copies=[r for r in copies if r['start']>=left and r['end']<=right]
 kernel_stats=aggregate(kernels);api_stats=aggregate(api);gpu=[dict(r) for r in c.execute('select name,uuid,computeMajor,computeMinor from TARGET_INFO_GPU where computeMajor>0')];assert any(g['computeMajor']==12 for g in gpu)
 result={'status':'warm_trace_evidence_verified','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files_verified':len(e['files']),'library_sha256':e['library_sha256'],'exact_warmup_cases':12,'exact_capture_cases':2,'warm_marker_interval_ns':[left,right],'full_trace_kernel_records':all_kernel_records,'kernel_records':len(kernels),'kernel_graph_node_records':sum(bool(k['graphNodeId']) for k in kernels),'api_records':len(api),'memory_copy_records':len(copies),'kernel_sum_duration_ns':sum(k['end']-k['start'] for k in kernels),'kernel_interval_union_ns':union_ns(kernels),'kernel_span_ns':max(k['end'] for k in kernels)-min(k['start'] for k in kernels),'kernels_by_name':kernel_stats,'apis_by_name':api_stats,'gpu':gpu,'two_x_established':False,'scope':'Warm correctness-checked text/image MTP trace selected by unique35892/35908byte HtoD markers. Cold/build records excluded from this ranking. Instrumented durations include profiler overhead; kernel sums may overlap and host API sums are not additive with GPU time. Node names identify collected operations, not proof of an optimization or uninstrumented model speed. No HTTP, true CUDA batching, full-context, broad-quality, repeated/tuned reference or2x admission.'}
 c.close()
(out/'warm-mtp-cuda-trace-2026-attempt1-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps({k:v for k,v in result.items() if k not in ('kernels_by_name','apis_by_name')},indent=2));print('TOP_KERNELS',json.dumps(kernel_stats[:15],indent=2));print('TOP_APIS',json.dumps(api_stats[:12],indent=2))
