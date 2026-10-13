import pathlib,json,hashlib,tarfile,numpy as np
r=pathlib.Path('/Users/zsdx/garnet-metal-investigation');jobs={'decode-accumulators-20261013':272,'decode-accumulators-head-confirm-20261013':6};proof={'jobs':{},'omitted_files':{}}
for name,count in jobs.items():
 p=r/'logs'/name;assert json.loads((p/'exit.json').read_text())['exit_code']==0
 summary=json.loads((p/'jobs/summary.json').read_text());assert len(summary['checks'])==count and len(summary['results'])==count
 assert all(v['fp64_reference_passed'] and v['padding_preserved'] and v['worst_forward_error_ratio']<=1 for v in summary['checks'])
 statuses=json.loads((p/'jobs/statuses.json').read_text());assert all(v['exit_code']==0 and v['interruption'] is None for v in statuses)
 assert all(json.loads((p/'jobs'/(v['name']+'-monitor.json')).read_text()) for v in statuses if v['name'].endswith('-gpu'))
 metadata=json.loads((p/'jobs/metadata.json').read_text())
 for basename,sha in metadata['source_sha256'].items():
  f=r/'source/src/backends/metal/metal_context.mm' if basename=='metal_context.mm' else r/'source/test2026/metal'/basename
  assert hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==sha
 # Establish that the independent FP64 reference and its bound are finite.
 audits=[]
 for manifest in (p/'jobs').glob('*-assets/manifest.json'):
  m=json.loads(manifest.read_text());assert hashlib.file_digest(pathlib.Path(m['checkpoint']).open('rb'),'sha256').hexdigest()==m['checkpoint_sha256']
  for case in m['cases']:
   f=manifest.parent/(case['name']+'.bin');assert hashlib.file_digest(f.open('rb'),'sha256').hexdigest()==case['sha256']
   bits=np.fromfile(f,dtype='<u2');assert len(bits)==case['columns']*case['inner']
   values=(bits.astype(np.uint32)<<16).view(np.float32);assert np.isfinite(values).all()
   largest=float(np.max(np.abs(values)));assert largest*case['inner']*.3<1e30
   audits.append({'model_assets':manifest.parent.name,'matrix':case['name'],'finite':True,'largest_absolute_weight':largest,'max_dot_absolute_bound':largest*case['inner']*.3})
 proof['jobs'][name]={'checks':count,'worst_forward_error_ratio':max(v['worst_forward_error_ratio'] for v in summary['checks']),'finite_asset_audits':audits}
out=r/'decode-accumulators-evidence.tar.gz'
with tarfile.open(out,'w:gz') as a:
 for name in jobs:
  for f in sorted((r/'logs'/name).rglob('*')):
   if not f.is_file():continue
   if f.suffix=='.bin' or f.name in ('decode_accumulators_gpu','decode_accumulators_head_gpu'):
    proof['omitted_files'][str(f.relative_to(r))]={'size':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()}
   else:a.add(f,arcname='completed/'+str(f.relative_to(r/'logs')))
 for name in ('decode_accumulators_gpu.mm','decode_accumulators_head_gpu.mm','make_decode_accumulators_shader.py','run_decode_accumulators_probe.py','run_decode_accumulators_head_probe.py','prepare_decode_tile_probe.py'):
  a.add(r/'source/test2026/metal'/name,arcname='source/'+name)
 a.add(r/'source/src/backends/metal/metal_context.mm',arcname='source/metal_context.mm')
 for name in ('decode-accumulators-source.tar.gz','decode-accumulators-source-manifest.json','decode-accumulators-head-source.tar.gz','decode-accumulators-head-source-manifest.json'):
  a.add(r/name,arcname='source-history/'+name)
 p=r/'decode-accumulators-evidence-proof.json';p.write_text(json.dumps(proof,indent=2));a.add(p,arcname='evidence-proof.json')
 a.add(pathlib.Path(__file__),arcname='source/archive_decode_accumulators.py')
print('Archive SHA256',hashlib.file_digest(out.open('rb'),'sha256').hexdigest())
