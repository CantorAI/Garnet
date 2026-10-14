"""Audit unchanged numerical scope, repaired sources and actual head use."""
from pathlib import Path
source=Path(__file__).with_name('verify_packed_prefill_norm_native.py').read_text()
start=source.index("assert proof['jobs']['packed-prefill-norm-resume-checks-20261014']");end=source.index('staging=base/',start)
source=source[:start]+"assert proof['jobs']['single-coder-head-checks-20261014']['exit_code']==1\nassert proof['jobs']['single-coder-head-repair-checks-20261014']['exit_code']==proof['jobs']['single-coder-head-native-20261014']['exit_code']==0\n"+source[end:]
source=source.replace('packed-prefill-norm-staging','single-coder-head-repair-staging').replace('packed-prefill-norm-original-source-20261014','single-coder-head-repair-original-source-20261014').replace('packed-prefill-norm-resume-checks-20261014','single-coder-head-repair-checks-20261014').replace('packed-prefill-norm-resume-native-20261014','single-coder-head-native-20261014')
start=source.index("prior=base/'logs/packed-prefill-norm-checks-20261014");end=source.index('for stage in stages[3:]:',start)
source=source[:start]+source[end:];source=source.replace('for stage in stages[3:]:','for stage in stages:')
source=source.replace("'METAL_PACKED_PREFILL_NORM' in meta['options']","'METAL_SINGLE_CODER_HEAD' in meta['options'] and 'METAL_GLOBAL_HALF_ALPHA' in meta['options']")
source=source.replace("'METAL_JOINED_PREFILL','METAL_RESIDUAL_NORM')","'METAL_JOINED_PREFILL','METAL_RESIDUAL_NORM','METAL_PACKED_PREFILL_NORM','METAL_JOINED_GLOBAL_MLP')")
source=source.replace("assert r['packed_prefill_norm_enabled']==candidate","assert r['single_coder_head_enabled']==candidate and r['global_half_alpha_enabled'] and not r['packed_prefill_norm_enabled'] and not r['joined_global_mlp_enabled']")
source=source.replace("assert r['prefill_profile'].get('packed_prefill_norm_dispatches',0)==(layers*len(r['prefill_chunk_plan']) if candidate else 0)","assert r['prefill_profile'].get('global_half_alpha_dispatches',0)>0\n  counts=r['single_coder_head_counts'];assert counts\n  if p['model']!='Qwen3.5-0.8B':assert len(counts)==r['decode_steps_after_first']+1\n  expected_count=1 if candidate and (p['model'],p['batch'])==('Qwen2.5-Coder-0.5B-Instruct',1) else 0\n  assert all(c==expected_count for c in counts)")
source=source.replace('all_actual_prefill_counters_match','all_actual_single_coder_head_counts_checked').replace('Packed prefill versus identical attention; no fresh vLLM comparison','Single coder head versus global alpha; no fresh vLLM comparison')
source=source.replace("meta=json.loads((checks/'metadata.json').read_text())", "assert 'Single coder head full-bit/F64/argmax, batch-two fallback, reset and abort PASS' in (checks/'CTest-LastTest.log').read_text()\nmeta=json.loads((checks/'metadata.json').read_text())")
source=source.replace('profiles.append(dict(**p,distinct_timing_phases=extra))', "profiles.append(dict(**p,distinct_timing_phases=extra,actual_single_coder_head_counts={policy:[r['single_coder_head_counts'] for r in records if r['build_policy']==policy] for policy in ('control','candidate')}))")
repair_audit="""
first=base/'single-coder-head-staging'
first_manifest=json.loads((first/'manifest.json').read_text())
assert first_manifest==original
first_original=json.loads((first/'original.json').read_text())
for name,sha in first_original.items():assert digest(base/'logs/single-coder-head-original-source-20261014'/name)==sha
unit='test2026/metal/single_coder_head_test.cpp'
with tarfile.open(first/'source.tar.gz') as archive:
 assert {m.name for m in archive.getmembers()}==set(first_manifest)
 for m in archive.getmembers():
  assert m.isfile();raw=archive.extractfile(m).read();assert hashlib.sha256(raw).hexdigest()==first_manifest[m.name]
  if m.name!=unit:assert manifest[m.name]==first_manifest[m.name]
  else:
   before=b'        auto w=gpu.UploadBF16(small,sizeof(small),2,2),a=gpu.Upload({1,2},1,2),o=gpu.Allocate(1,2);'
   after=b'        auto w=gpu.UploadBF16(small,sizeof(small),2,2);\\n        auto a=gpu.Upload({1,2},1,2),o=gpu.Allocate(1,2);'
   assert raw.count(before)==1 and raw.replace(before,after)==(base/'source'/unit).read_bytes()
failed=base/'logs/single-coder-head-checks-20261014/jobs/attempts/build/case0-attempt0/build.log'
assert "'auto' deduced" in failed.read_text() and 'MetalBF16Weight' in failed.read_text() and 'MetalMatrix' in failed.read_text()
"""
source=source.replace("checks=base/'logs/single-coder-head-repair-checks-20261014/jobs'",repair_audit+"\nchecks=base/'logs/single-coder-head-repair-checks-20261014/jobs'")
exec(compile(source,str(Path(__file__)),'exec'))
