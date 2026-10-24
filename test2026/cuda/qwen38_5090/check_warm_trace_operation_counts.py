from pathlib import Path
import hashlib,json,zipfile
out=Path(__file__).parent
v=json.loads((out/'warm-mtp-cuda-trace-2026-attempt1-verified.json').read_text())
archive=out/'warm-mtp-cuda-trace-2026-attempt1-evidence.zip'
assert hashlib.sha256(archive.read_bytes()).hexdigest()==v['archive_sha256']
with zipfile.ZipFile(archive) as z,zipfile.ZipFile(out/'native-mtp3-decode-only-off-http-evidence.zip') as old:
 r=json.loads(z.read('results/warm-mtp-cuda-trace-2026-attempt1/native/report.json'));cases=[x for x in r['cases'] if x['phase']=='capture']
 prefill=0
 for c in cases:
  teacher=json.loads(old.read('results/shared-decode-only-off-http-diagnostic/'+c['key']+'-target-response.json'))['choices'][0]
  assert 0<len(teacher['prompt_token_ids'])<=128
  prefill+=1
m=[c['metrics'] for c in cases];target=sum(x['target_forward_calls'] for x in m)+prefill
mtp=sum(x[k] for x in m for k in ('mtp_prefill_calls','mtp_draft_calls','mtp_correction_calls'))
heads=sum(x['target_head_evaluations']+x['mtp_head_calls'] for x in m)+prefill
prefix=sum(x['target_prefix_replays'] for x in m)
# Pinned 64-layer joined model:128 body NVFP4 projections,48 GDN layers
# with2 uncompressed a/b projections; MTP chunk has6 BF16 projections.
# Include actual prefill heads, excluded by generation-only head counters.
expected={'nvfp4_matmul':128*target+heads,'nvfp4_quantizer':128*target+heads,'bf16_small_tile':96*target+6*mtp,'gdn_recurrence':48*(target+prefix)}
selectors={'nvfp4_matmul':lambda n:n.startswith('cutlass3x_sm120_bstensorop_'),'nvfp4_quantizer':lambda n:'::QuantizeActivation(' in n,'bf16_small_tile':lambda n:'cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8' in n,'gdn_recurrence':lambda n:'::recurrence(' in n}
actual={key:sum(row['count'] for row in v['kernels_by_name'] if match(row['name'])) for key,match in selectors.items()}
assert actual==expected,(actual,expected)
result={'status':'selected_operation_count_audit_passed','trace_archive_sha256':v['archive_sha256'],'target_calls_including_prefill':target,'mtp_body_calls':mtp,'head_calls_including_prefill':heads,'prefix_replays':prefix,'actual':actual,'expected':expected,'scope':'Partial trace-coverage cross-check for four pinned operation families against complete capture-case counters and verified teacher prompt lengths. Structural projection counts are source-derived assumptions; does not prove completeness of all CUPTI operations, timing accuracy or speedup.'}
(out/'warm-mtp-trace-operation-counts.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
