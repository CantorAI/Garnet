from pathlib import Path
import json,hashlib,collections
d=Path('/Users/shawn/CantorAI/Garnet/results/20261016/worldsense-m5-primary-blocks');assert d.name=='worldsense-m5-primary-blocks'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((d/'file-manifest.json').read_text());actual={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'};assert actual==set(manifest)
for rel,h in manifest.items():assert sha(d/rel)==h,rel
selection=json.loads((d/'matrix-selection.json').read_text());assert sha(d/'source-inventory.json')==selection['inventory_sha256']
inv=json.loads((d/'source-inventory.json').read_text());bf16={r['buffer_id']for r in inv['references']if r['representation']=='bf16'};assert bf16=={x['inventory_buffer_id']for x in selection['matrices']}
a={(x['model'],x['tensor']):x for x in selection['matrices']};report=json.loads((d/'scan-results.json').read_text());b={(x['model'],x['tensor']):x for x in report['matrices']};assert a.keys()==b.keys()and len(a)==446
totals=collections.Counter()
for key,row in b.items():
 assert row['all_original_BF16_bits_exact']and row['original_bytes']==a[key]['original_buffer_length_bytes']==row['rows']*row['inner']*2
 assert row['encoded_payload_bytes']==row['compact_blocks']*768+row['raw_blocks']*1024
 assert row['metadata_bytes']==row['blocks']*8 and row['blocks']==row['compact_blocks']+row['raw_blocks']
 for k in ['original_bytes','retained_primary_bytes','page_rounded_retained_estimate','blocks','compact_blocks']:totals[k]+=row[k]
 for k in ['raw_tensor_sha256','encoded_payload_stream_sha256','metadata_stream_sha256']:assert len(row[k])==64 and all(c in '0123456789abcdef'for c in row[k])
assert all(report['totals'][k]==v for k,v in totals.items())
assert report['status']=='cpu_feasibility_complete_exact_bits_not_GPU_validated'and totals['original_bytes']==11850481664 and totals['retained_primary_bytes']==9078100992
status=json.loads((d/'execution-status.json').read_text());assert status['returncode']==0 and status['dependency_identities_still_match']
for rel,h in json.loads((d/'cpu-source-manifest.json').read_text()).items():assert sha(d/'source/cpu-source'/rel)==h
print(json.dumps({'verified_sha256_files':len(manifest),'complete_selected_matrices':len(a),'original_words_covered':totals['original_bytes']//2,'scope':'Archive integrity, selection/count arithmetic and consumed native-source identity. Not independent GPU/FP32 equivalence or performance verification.'}))
