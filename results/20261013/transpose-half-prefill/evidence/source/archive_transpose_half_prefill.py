import pathlib,json,hashlib,tarfile
r=pathlib.Path('/Users/zsdx/garnet-metal-investigation');names=['transpose-half-prefill-20261013','transpose-half-prefill-confirm-20261013'];proof={'jobs':{},'omitted_files':{}}
for name in names:
 p=r/'logs'/name;assert json.loads((p/'exit.json').read_text())['exit_code']==0
 statuses=json.loads((p/'jobs/statuses.json').read_text());assert len(statuses)==7 and all(v['exit_code']==0 and v['interruption'] is None for v in statuses)
 summary=json.loads((p/'jobs/summary.json').read_text());assert len(summary['checks'])==216 and len(summary['excluded'])==1
 assert all(v['encoded_weight_bits_equal'] and v['padding_preserved'] and v['worst_ratio']<=1 for v in summary['checks'])
 metadata=json.loads((p/'jobs/metadata.json').read_text())
 for n,sha in metadata['source_sha256'].items():assert hashlib.file_digest((r/'source'/n).open('rb'),'sha256').hexdigest()==sha
 for v in statuses:
  if v['name'].endswith('-gpu'):assert json.loads((p/'jobs'/(v['name']+'-monitor.json')).read_text())
 proof['jobs'][name]={'checks':len(summary['checks']),'excluded':summary['excluded'],'worst_ratio':max(v['worst_ratio'] for v in summary['checks'])}
out=r/'transpose-half-prefill-evidence.tar.gz'
with tarfile.open(out,'w:gz') as a:
 for name in names:
  for f in sorted((r/'logs'/name).rglob('*')):
   if not f.is_file():continue
   if f.suffix=='.bin' or f.name=='transpose_half_prefill_probe':proof['omitted_files'][str(f.relative_to(r))]={'size':f.stat().st_size,'sha256':hashlib.file_digest(f.open('rb'),'sha256').hexdigest()}
   else:a.add(f,arcname='completed/'+str(f.relative_to(r/'logs')))
 for name in ('transpose_half_prefill_probe.mm','run_transpose_half_prefill_probe.py','prepare_prefill_tile_probe.py','audit_exact_half_qkv_admission.py'):a.add(r/'source/test2026/metal'/name,arcname='source/'+name)
 a.add(r/'source/src/backends/metal/exact_half_codec.h',arcname='source/exact_half_codec.h')
 for name in ('transpose-half-prefill-source.tar.gz','transpose-half-prefill-manifest.json'):a.add(r/name,arcname='source-history/'+name)
 density=r/'logs/exact-half-qkv-exception-density-20261013/summary.json';a.add(density,arcname='diagnostic/qkv-exception-density.json')
 verified=r/'logs/exact-half-qkv-admission-audit-20261013/summary.json';a.add(verified,arcname='diagnostic/qkv-admission-audit.json')
 p=r/'transpose-half-prefill-evidence-proof.json';p.write_text(json.dumps(proof,indent=2));a.add(p,arcname='evidence-proof.json');a.add(pathlib.Path(__file__),arcname='source/archive_transpose_half_prefill.py')
print('Archive SHA256',hashlib.file_digest(out.open('rb'),'sha256').hexdigest())
