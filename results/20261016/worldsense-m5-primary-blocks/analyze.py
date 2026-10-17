from pathlib import Path
import json,hashlib,collections
w=Path(__file__).resolve().parent;d=json.loads((w/'scan-results.json').read_text());selection=json.loads((w/'matrix-selection.json').read_text());state=json.loads((w/'execution-status.json').read_text());freeze=json.loads((w/'execution-freeze.json').read_text())
assert state['returncode']==0 and state['dependency_identities_still_match'] and state['executable_sha256']==freeze['executable_sha256'] and d['status']=='cpu_feasibility_complete_exact_bits_not_GPU_validated'
assert d['selftest']['finite_BF16_words_exhaustively_checked']==65280 and d['selftest']['mixed_variable_blocks']==54 and d['selftest']['bad_offset_nonfinite_shape_and_decode_bounds_rejected']
a={(x['model'],x['tensor']):x for x in selection['matrices']};b={(x['model'],x['tensor']):x for x in d['matrices']};assert a.keys()==b.keys()and len(a)==446
models={};totals=collections.Counter()
for key,row in b.items():
 ref=a[key];assert row['rows']==ref['rows']and row['inner']==ref['inner']and row['original_bytes']==ref['original_buffer_length_bytes']and row['all_original_BF16_bits_exact']
 assert row['blocks']==row['rows']//4*(row['inner']//128) and row['compact_blocks']+row['raw_blocks']==row['blocks'] and row['encoded_payload_bytes']==row['compact_blocks']*768+row['raw_blocks']*1024 and row['metadata_bytes']==row['blocks']*8
 assert row['retained_primary_bytes']==(row['encoded_payload_bytes']+row['metadata_bytes']if row['admitted_by_bytes_only']else row['original_bytes'])
 bucket=models.setdefault(row['model'],collections.Counter())
 for target in [bucket,totals]:
  target['matrices']+=1
  for k in ['original_bytes','original_Metal_allocated_bytes','retained_primary_bytes','page_rounded_retained_estimate','blocks','compact_blocks','metadata_bytes']:target[k]+=row[k]
  target['words_reconstructed']+=row['original_bytes']//2
  target['admitted']+=row['admitted_by_bytes_only']
  if row['rows']>256:
   target['planned_large_row_matrices']+=1;target['planned_large_row_original_bytes']+=row['original_bytes'];target['planned_large_row_retained_bytes']+=row['retained_primary_bytes'];target['planned_large_row_page_estimate']+=row['page_rounded_retained_estimate']
  else:target['small_row_original_bytes']+=row['original_bytes']
assert totals['original_bytes']==11850481664 and totals['retained_primary_bytes']==9078100992 and totals['page_rounded_retained_estimate']==9083469824
report={'status':'exact_CPU_codec_feasible_GPU_implementation_unproven','models':models,'totals':totals,'logical_BF16_byte_reduction':1-totals['retained_primary_bytes']/totals['original_bytes'],'compact_fraction':totals['compact_blocks']/totals['blocks'],'scope':'Full actual WorldSense raw-BF16-buffer coverage, not total model memory, GPU allocations or inference speed. Reconstructed words are integer BF16 bit patterns; no claim of GPU FP32 accumulation correctness yet. Small rows may retain raw format in initial GPU implementation; separately counted.'}
(w/'feasibility-analysis.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
