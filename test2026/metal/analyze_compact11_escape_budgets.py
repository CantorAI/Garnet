"""Correct descriptive packet budgets for common raw-fallback metadata; no runtime prediction."""
from pathlib import Path
import json,sys
base=Path(sys.argv[1]).resolve();audit=json.loads((base/'verification.json').read_text())
assert audit['fixture_matrices']==29 and audit['eligible_matrices']==20 and audit['captured_exact_expression_values']==64268800
frozen=json.loads((base/'archive-verification.json').read_text())
assert frozen['archive_sha256']=='94fba9ef5ba375c688e574b079a04fddf0ace612c71224c130c17c869a0af4d3'
models=[]
for r in audit['models']:
    t=r['totals'];fallback=t['logical_raw_fallback_read_bytes'];extra=t['raw_fallback_blocks']*4
    control=t['retained_compact_payload_metadata_bytes']+fallback+extra
    ideal=t['hypothetical_ideal_packet_bytes']+fallback+extra
    ranked=t['hypothetical_ranked_aligned_packet_bytes']+fallback+extra
    models.append(dict(model=r['model'],eligible_matrices=t['eligible_matrices'],blocks=t['blocks'],
        retained_packet_bytes=control,ideal_packet_bytes=ideal,ranked_packet_bytes=ranked,
        ideal_packet_fraction=ideal/control,ranked_packet_fraction=ranked/control,
        common_raw_fallback_metadata_bytes=extra))
result=dict(models=models,correction='Initial CPU ratios omit the identical 4-byte metadata read for each raw-fallback block in both arms; retain those original bytes and add the common term here.',
    ranked_gpu_probe_admitted=False,ideal_gpu_probe_admitted=False,
    decision='Prioritization after descriptive CPU results, not a predeclared speed gate: ranked/aligned estimated payload saving is small; ideal assumes free escape rank/address lookup that is not implemented. No runtime candidate selected.',
    scope='Static packet model only, including common raw-fallback metadata. Neither GPU traffic/cache behavior nor process RSS, numerical GPU output or model speed is measured. Variable offsets/rank lookup are unimplemented; original master buffers remain.',full_goal_met=False)
(base/'budget-analysis.json').write_bytes((json.dumps(result,indent=2)+'\n').encode());print(json.dumps(result))
