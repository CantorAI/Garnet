"""Check report inventory, local citations and documented benchmark numbers.

This verifies documentation consistency, not numerical correctness of experiments.
Run the separate archived experiment verifiers for the latter.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import re
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / 'docs/research'
OUT = DOCS / 'metal-method-evidence'


def main():
    index = json.loads((OUT / 'index.json').read_text(encoding='utf-8'))
    expected = {p.relative_to(ROOT).as_posix() for base in
                (ROOT / 'results', ROOT / 'test2026/metal/results')
                for p in base.rglob('README.md')}
    assert expected == {r['report_path'] for r in index['reports']}
    verified = 0
    paths = [DOCS / 'metal_optimization_research_log.md', OUT / 'INDEX.md']
    for record in index['reports']:
        p = ROOT / record['report_path']
        raw = p.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == record['report_sha256'], p
        assert raw.decode('utf-8-sig') == record['report_text'], p
        for item in record['verification_records']:
            assert hashlib.sha256((ROOT / item['path']).read_bytes()).hexdigest() == item['sha256']
            verified += 1
    # Validate this curated journal/index and newly added reports, not arbitrary
    # historical report links which may reference remote-only files.
    paths += [ROOT / 'results/20261015' / name / 'README.md' for name in
              ('dependent-mlp-graph', 'stage-rounded-mlp-graph', 'mlp-prefill-native',
               'coalesced-eight-body','mlp-command-buffer-primitive','mlp-command-buffer-native','eight-fma-body',
               'two-body','two-matrix-probe','shape-selected-two-native','mlp-bf16-feed-primitive','four-threadgroup-probe',
               'ndarray-matrix-api','ndarray-prefill','ndarray-native','vector-vision-context')]
    paths += [ROOT / 'results/20261016' / name / 'README.md' for name in
              ('ndarray-weight-admission','ndarray-selective-prefill','ndarray-selective-native','compact11-escape-cpu','vector-vision-calibrated','joined-union-half-admission','prepared-union-mlp','qkv-union-half-admission','prepared-union-qkv','prepared-qkv-native','prepared-half-inverse','prepared-union-qv','prepared-union-qv-views','paired-norm-rope','paired-norm-rope-chain','paired-norm-rope-native','delta-value-tiles-addresses','delta-value-tiles-gpu')]
    links = 0
    for p in paths:
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', p.read_text(encoding='utf-8')):
            if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*:', target) or target.startswith('#'):
                continue
            target = unquote(target.split('#')[0].strip('<>'))
            assert (p.parent / target).resolve().exists(), (p, target)
            links += 1
    # Reconcile both tables in the newest matched report, including ratios.
    p = ROOT / 'results/20261015/coalesced-body-vllm'
    data = json.loads((p / 'remaining-budgets.json').read_text())
    rows = [line.split('|')[1:-1] for line in (p / 'README.md').read_text().splitlines()
            if line.startswith('| Qwen')]
    profiles = [r for trial in data['trials'] for r in trial['profiles']]
    assert len(rows) == len(profiles) == 12
    numbers = 0
    for cells, r in zip(rows, profiles):
        assert cells[0].strip() == f"{r['model']} / {r['batch']}"
        g, v = r['means']['garnet'], r['means']['vllm-metal']
        expect = [[g['decode_tps'], v['decode_tps']], [g['ttft_ms'], v['ttft_ms']],
                  [r['decode_ratio']], [r['ttft_fraction']]]
        for cell, values in zip(cells[1:], expect):
            assert cell.strip() == ' / '.join(f'{x:.4f}' for x in values)
            numbers += len(values)
    # Component report tables are generated from the admitted verifier profiles.
    component_numbers = 0
    for name in ('dependent-mlp-graph', 'stage-rounded-mlp-graph'):
        p = ROOT / 'results/20261015' / name
        data = json.loads((p / 'verification.json').read_text())
        rows = [line.split('|')[1:-1] for line in (p / 'README.md').read_text().splitlines()
                if line.startswith('| Qwen')]
        assert len(rows) == len(data['profiles'])
        for cells, r in zip(rows, data['profiles']):
            assert cells[0].strip() == f"{r['model']} / {r['batch']}"
            assert cells[1].strip() == {1:'placeholder', 2:'constant'}[r['policy']]
            for cell, value in zip(cells[2:], [r['wall_ms']['control'], r['wall_ms']['candidate'], r['wall_fraction']]):
                assert cell.strip() == f'{value:.6f}'
                component_numbers += 1
    # Rejected full-native integrations: preserve slower timings and memory.
    native_numbers=0
    for name in ('mlp-prefill-native','mlp-command-buffer-native'):
        p=ROOT/'results/20261015'/name
        native=json.loads((p/'comparison/verification.json').read_text())['profiles']
        rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
        assert len(rows)==len(native)==6
        for cells,r in zip(rows,native):
            label=r['model']
            if name=='mlp-prefill-native':
                label={'Qwen2.5-Coder-0.5B-Instruct':'Qwen2.5-Coder-0.5B','Qwen3.5-0.8B':'Qwen3.5-0.8B VL'}.get(label,label)
            assert cells[0].strip()==label and int(cells[1])==r['batch']
            means=r['means'];first=r['first_request_after_initialization'];rss=r['peak_individual_rss_bytes']
            formatted=[
                ' / '.join(f"{means[k]['ttft_ms']:.3f}" for k in ('control','candidate')),
                f"{r['decode_ratio']:.3f}",
                ' / '.join(f"{first[k]['ttft_ms']:.3f}" for k in ('control','candidate')),
                f"{(rss['candidate']-rss['control'])/2**30:.3f}"]
            assert [c.strip() for c in cells[2:]]==formatted
            native_numbers+=6
    selected_native_numbers=0
    p=ROOT/'results/20261015/shape-selected-two-native'
    selected=json.loads((p/'comparison/verification.json').read_text())['profiles']
    decision=json.loads((p/'decision-analysis.json').read_text())['profiles']
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==len(selected)==len(decision)==6
    for cells,r,d in zip(rows,selected,decision):
        assert cells[0].strip()==r['model']==d['model'] and int(cells[1])==r['batch']==d['batch']
        assert int(cells[2])==d['selected_dispatches_per_decode_step']
        m=r['means'];formatted=[
            ' / '.join(f"{m[k]['decode_tps']:.4f}" for k in ('control','candidate')),
            ' / '.join(f"{m[k]['ttft_ms']:.4f}" for k in ('control','candidate')),
            f"{r['decode_ratio']:.6f}",f"{r['ttft_fraction']:.6f}"]
        assert [c.strip() for c in cells[3:]]==formatted
        selected_native_numbers+=7
    p=ROOT/'results/20261015/mlp-bf16-feed-primitive'
    bf16=json.loads((p/'component/verification.json').read_text())['profiles']
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==len(bf16)==6
    bf16_numbers=0
    for cells,r in zip(rows,bf16):
        assert cells[0].strip()==r['model'] and int(cells[1])==r['batch']
        assert cells[2].strip()=={1:'placeholder',2:'constant'}[r['policy']]
        assert [c.strip() for c in cells[3:]]==[f"{v:.6f}" for v in (r['wall_ms']['control'],r['wall_ms']['candidate'],r['wall_fraction'])]
        bf16_numbers+=3
    p=ROOT/'results/20261015/four-threadgroup-probe'
    calibration=json.loads((p/'confirmation-comparison.json').read_text())
    assert calibration['selected']==[]
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines()
          if line.startswith(('| initial |','| confirmation |'))]
    assert len(rows)==len(calibration['null_summary'])==2
    null_numbers=0
    for cells,r in zip(rows,calibration['null_summary']):
        assert cells[0].strip()==r['cohort']
        g,w=r['gpu_fractions'],r['wall_fractions']
        assert [c.strip() for c in cells[1:]]==[
            f"{g['minimum']:.6f} / {g['maximum']:.6f}",str(g['outside_five_percent']),
            f"{w['minimum']:.6f} / {w['maximum']:.6f}",str(w['outside_five_percent'])]
        null_numbers+=6
    p=ROOT/'results/20261015/ndarray-prefill'
    cohorts=[json.loads((p/n/'verification.json').read_text()) for n in ('repaired','confirmation')]
    decision=json.loads((p/'decision-analysis.json').read_text())
    assert decision['confirmation_complete'] and decision['qualified']==[dict(model='Qwen3.5-0.8B',batch=1,policy=2)]
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==len(cohorts[0]['profiles'])==len(cohorts[1]['profiles'])==12
    ndarray_numbers=0
    for cells,a,b in zip(rows,cohorts[0]['profiles'],cohorts[1]['profiles']):
        assert (a['model'],a['batch'],a['policy'])==(b['model'],b['batch'],b['policy'])
        assert cells[0].strip()==a['model'] and int(cells[1])==a['batch'] and cells[2].strip()=={1:'mixed',2:'F32',3:'null'}[a['policy']]
        assert [c.strip() for c in cells[3:]]==[f'{v:.6f}' for v in (a['gpu_fraction'],a['wall_fraction'],b['gpu_fraction'],b['wall_fraction'])]
        ndarray_numbers+=4
    p=ROOT/'results/20261015/ndarray-native'
    native_nd=json.loads((p/'comparison/verification.json').read_text())
    nd_decision=json.loads((p/'decision-analysis.json').read_text())
    assert native_nd['full_output_requests']==72 and native_nd['actual_ndarray_prefill_routes']
    assert not nd_decision['promote'] and not nd_decision['request_confirmation'] and not nd_decision['fresh_vllm_run']
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==len(native_nd['profiles'])==len(nd_decision['profiles'])==6
    ndarray_native_numbers=0
    for cells,r,d in zip(rows,native_nd['profiles'],nd_decision['profiles']):
        assert cells[0].strip()==r['model']==d['model'] and int(cells[1])==r['batch']==d['batch']
        m=r['means'];f=r['first_request_after_initialization'];rss=r['peak_individual_rss_bytes']
        formatted=[str(d['ndarray_prefill_calls']),
            ' / '.join(f"{m[k]['decode_tps']:.4f}" for k in ('control','candidate')),
            ' / '.join(f"{m[k]['ttft_ms']:.4f}" for k in ('control','candidate')),
            f"{r['decode_ratio']:.6f}",f"{r['ttft_fraction']:.6f}",
            ' / '.join(f"{f[k]['ttft_ms']:.3f}" for k in ('control','candidate')),
            f"{(rss['candidate']-rss['control'])/2**30:.3f}"]
        assert [c.strip() for c in cells[2:]]==formatted
        assert d['aggregate_decode_tps']=={k:v['decode_tps']*r['batch'] for k,v in m.items()}
        ndarray_native_numbers+=10
    initrows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| init Qwen')]
    assert len(initrows)==6
    for cells,r in zip(initrows,native_nd['profiles']):
        assert cells[0].strip()=='init '+r['model'] and int(cells[1])==r['batch']
        f=r['first_request_after_initialization']
        assert [c.strip() for c in cells[2:]]==[f"{f[k]['initialization_s']:.6f}" for k in ('control','candidate')]
        ndarray_native_numbers+=2
    p=ROOT/'results/20261016/ndarray-weight-admission'
    admission=json.loads((p/'verification.json').read_text())
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==len(admission['models'])==3 and admission['body_matrices']==498
    admission_numbers=0
    for cells,r in zip(rows,admission['models']):
        assert cells[0].strip()==r['model']
        observed=[int(c.strip().replace(',','').removesuffix(' bytes')) for c in cells[1:]]
        assert observed==[r['matrices'],r['global_half_ready'],r['exact_half_ready'],len(r['no_exact_half']),r['no_exact_f32_cache_bytes']]
        admission_numbers+=5
    p=ROOT/'results/20261016/ndarray-selective-prefill'
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith(('| Initial |','| Repeat |'))]
    decision=json.loads((p/'decision-repeat.json').read_text())
    assert decision['native_integration_admitted'] and not decision['promote'] and len(rows)==4
    selective_numbers=0
    for i,(cells,record) in enumerate(zip(rows,[r for t in decision['trials'] for r in t['profiles']])):
        label='initial' if i<2 else 'repeat';audit=json.loads((p/label/'verification.json').read_text())
        profile=next(v for v in audit['profiles'] if v['model']=='Qwen3.5-0.8B' and v['batch']==record['batch'] and v['policy']==2)
        assert cells[0].strip()==label.capitalize() and int(cells[1])==record['batch']
        formatted=[' / '.join(f"{profile['means'][k][metric+'_ms']:.6f}" for k in ('control','candidate')) for metric in ('gpu','wall')]
        formatted += [f"{record[k]:.6f}" for k in ('gpu_fraction','wall_fraction','null_gpu_fraction','null_wall_fraction')]
        assert [c.strip() for c in cells[2:]]==formatted
        selective_numbers+=8
    p=ROOT/'results/20261016/compact11-escape-cpu'
    compact=json.loads((p/'budget-analysis.json').read_text())
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==len(compact['models'])==3 and not compact['ranked_gpu_probe_admitted'] and not compact['full_goal_met']
    compact_numbers=0
    for cells,r in zip(rows,compact['models']):
        assert cells[0].strip()==r['model']
        assert [int(c.strip().replace(',','')) for c in cells[1:4]]==[r['eligible_matrices'],r['blocks'],r['retained_packet_bytes']]
        assert [c.strip() for c in cells[4:]]==[f"{r[k]:.6f}" for k in ('ideal_packet_fraction','ranked_packet_fraction')]
        compact_numbers+=5
    guard=json.loads((ROOT/'results/20261016/ndarray-selective-native/profile-guard-preflight.json').read_text())
    assert guard['observed_control_requests']==len(guard['checked'])==36 and not guard['full_goal_met']
    p=ROOT/'results/20261016/ndarray-selective-native'
    decision=json.loads((p/'comparison/decision-initial.json').read_text())
    audit=json.loads((p/'comparison/initial/verification.json').read_text())
    assert audit['full_output_requests']==72 and audit['actual_ndarray_prefill_routes']
    assert not decision['repeat_admitted'] and not decision['promote'] and not decision['full_goal_met']
    profiles=decision['trials'][0]['profiles'];text=(p/'README.md').read_text()
    rows=[line.split('|')[1:-1] for line in text.splitlines() if line.startswith('| Qwen')]
    assert len(rows)==len(profiles)==6;selective_native_numbers=0
    for cells,r in zip(rows,profiles):
        assert cells[0].strip()==r['model'] and int(cells[1])==r['batch']
        table_values=[' / '.join(f"{r['means'][k][metric]:.4f}" for k in ('control','candidate')) for metric in ('decode_tps','ttft_ms')]
        table_values += [f"{r[k]:.6f}" for k in ('decode_ratio','ttft_fraction')]
        table_values += [f"{r['added_mean_worker_peak_rss_gib']*1024:.3f}"]
        assert [c.strip() for c in cells[2:]]==table_values
        selective_native_numbers+=7
    rows=[line.split('|')[1:-1] for line in text.splitlines() if line.startswith('| init Qwen')];assert len(rows)==6
    for cells,r in zip(rows,profiles):
        assert cells[0].strip()=='init '+r['model'] and int(cells[1])==r['batch']
        first=r['first_request_after_initialization']
        table_values=[' / '.join(f"{first[k]['initialization_s']:.6f}" for k in ('control','candidate')),
                  ' / '.join(f"{first[k]['ttft_ms']:.3f}" for k in ('control','candidate'))]
        assert [c.strip() for c in cells[2:]]==table_values;selective_native_numbers+=4
    p=ROOT/'results/20261015/vector-vision-context'
    decision=json.loads((p/'decision-initial.json').read_text());assert not decision['repeat_admitted'] and not decision['qualified']
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith(('| 2 |','| 4 |'))]
    groups=[(c['vector_width'],s) for c in decision['trials'][0]['candidates'] for s in c['serving']]
    assert len(rows)==len(groups)==4;vector_initial_numbers=0
    for cells,(width,r) in zip(rows,groups):
        assert int(cells[0])==width and int(cells[1])==r['batch']
        assert [c.strip() for c in cells[2:]]==[f"{r[k]:.6f}" for k in ('gpu_fraction','wall_fraction','null_gpu_fraction','null_wall_fraction')]
        vector_initial_numbers+=4
    p=ROOT/'results/20261016/vector-vision-calibrated'
    calibration=json.loads((p/'calibration-decision.json').read_text());assert calibration['candidate_probe_admitted'] and not calibration['full_goal_met']
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| calibration-')]
    groups=[(t['label'],r) for t in calibration['trials'] for r in t['serving']];assert len(rows)==len(groups)==4
    vector_calibration_numbers=0
    for cells,(label,r) in zip(rows,groups):
        assert cells[0].strip()==label and int(cells[1])==r['batch']
        assert [c.strip() for c in cells[2:]]==[f"{r[k]:.6f}" for k in ('gpu_fraction','wall_fraction')]
        vector_calibration_numbers+=2
    vector_candidate=json.loads((p/'decision-repeat.json').read_text());assert vector_candidate['repeat_admitted'] and not vector_candidate['promote']
    assert vector_candidate['native_integration_vector_width']==4 and len(vector_candidate['trials'])==2
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith(('| 2 |','| 4 |'))]
    groups=[(c['vector_width'],r) for t in vector_candidate['trials'] for c in t['candidates'] for r in c['serving']]
    assert len(rows)==len(groups)==8;vector_calibrated_numbers=0
    for cells,(width,r) in zip(rows,groups):
        assert int(cells[0])==width and int(cells[1])==r['batch']
        assert [c.strip() for c in cells[2:]]==[f"{r[k]:.6f}" for k in ('gpu_fraction','wall_fraction','null_gpu_fraction','null_wall_fraction')]
        vector_calibrated_numbers+=4
    p=ROOT/'results/20261016/vector-vision-native'
    vector_native=json.loads((p/'comparison/decision-initial.json').read_text())
    vector_audit=json.loads((p/'comparison/initial/verification.json').read_text())
    assert vector_native['repeat_admitted'] and not vector_native['native_confirmation_pass'] and not vector_native['promote']
    assert vector_audit['full_output_requests']==72 and vector_audit['actual_vector_vision_routes'] and vector_audit['audit_path_repair_only']
    assert vector_audit['regenerated_native_comparators']==6 and vector_audit['distinct_expanded_script_files']==3
    assert json.loads((p/'validation/verification.json').read_text())['validation_stages']==30
    assert json.loads((p/'auditor-path-failure/archive-verification.json').read_text())['verified_files']==45
    vector_rows=vector_native['trials'][0]['profiles'];native_text=(p/'README.md').read_text()
    rows=[line.split('|')[1:-1] for line in native_text.splitlines() if line.startswith('| Native Qwen')]
    assert len(rows)==len(vector_rows)==6;vector_native_numbers=0
    for cells,r in zip(rows,vector_rows):
        assert cells[0].strip()=='Native '+r['model'] and int(cells[1])==r['batch']
        table_values=[' / '.join(f"{r['means'][k][metric]:.4f}" for k in ('control','candidate')) for metric in ('decode_tps','ttft_ms')]
        table_values += [f"{r[k]:.6f}" for k in ('decode_ratio','ttft_fraction')]+[f"{r['added_mean_worker_peak_rss_gib']*1024:.3f}"]
        assert [c.strip() for c in cells[2:]]==table_values;vector_native_numbers+=7
    rows=[line.split('|')[1:-1] for line in native_text.splitlines() if line.startswith('| First native Qwen')];assert len(rows)==6
    for cells,r in zip(rows,vector_rows):
        assert cells[0].strip()=='First native '+r['model'] and int(cells[1])==r['batch']
        first=r['first_request_after_initialization']
        assert [c.strip() for c in cells[2:]]==[' / '.join(f"{first[k]['initialization_s']:.6f}" for k in ('control','candidate')),' / '.join(f"{first[k]['ttft_ms']:.3f}" for k in ('control','candidate'))];vector_native_numbers+=4
    vector_repeat=json.loads((p/'comparison/decision-repeat.json').read_text())
    assert vector_repeat['native_confirmation_pass'] and not vector_repeat['promote'] and not vector_repeat['full_goal_met']
    repeat_audit=json.loads((p/'comparison/repeat/verification.json').read_text())
    assert repeat_audit['full_output_requests']==72 and repeat_audit['audit_path_repair_only'] and repeat_audit['actual_vector_vision_routes']
    repeat_profiles=vector_repeat['trials'][1]['profiles'];rows=[line.split('|')[1:-1] for line in native_text.splitlines() if line.startswith('| Repeat native Qwen')]
    assert len(rows)==len(repeat_profiles)==6;vector_repeat_numbers=0
    for cells,r in zip(rows,repeat_profiles):
        assert cells[0].strip()=='Repeat native '+r['model'] and int(cells[1])==r['batch']
        table_values=[' / '.join(f"{r['means'][k][metric]:.4f}" for k in ('control','candidate')) for metric in ('decode_tps','ttft_ms')]
        table_values += [f"{r[k]:.6f}" for k in ('decode_ratio','ttft_fraction')]+[f"{r['added_mean_worker_peak_rss_gib']*1024:.3f}"]
        assert [c.strip() for c in cells[2:]]==table_values;vector_repeat_numbers+=7
    rows=[line.split('|')[1:-1] for line in native_text.splitlines() if line.startswith('| Repeat first native Qwen')];assert len(rows)==6
    for cells,r in zip(rows,repeat_profiles):
        assert cells[0].strip()=='Repeat first native '+r['model'] and int(cells[1])==r['batch']
        first=r['first_request_after_initialization']
        assert [c.strip() for c in cells[2:]]==[' / '.join(f"{first[k]['initialization_s']:.6f}" for k in ('control','candidate')),' / '.join(f"{first[k]['ttft_ms']:.3f}" for k in ('control','candidate'))];vector_repeat_numbers+=4
    integration=json.loads((p/'branch-source-integration.json').read_text());assert integration['branch_source_integrated'] and integration['both_host_audits_pass'] and not integration['remote_default_library_changed'] and not integration['full_goal_met']
    assert len(integration['changes'])==8 and integration['measured_library_sha256']==vector_repeat['candidate_library_sha256']
    for name,info in integration['changes'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==info['after_sha256']
    compact_audit=json.loads((ROOT/'results/20261016/compact-fused-projection-groups/initial/verification.json').read_text())
    assert compact_audit['verified_files']==33 and compact_audit['group_batch_checks']==16 and compact_audit['validated_executions']==104093 and compact_audit['timed_executions']==6400
    compact_decision=json.loads((ROOT/'results/20261016/compact-fused-projection-groups/decision-initial.json').read_text());assert not compact_decision['repeat_admitted'] and not compact_decision['native_integration_groups']
    triplet_audit=json.loads((ROOT/'results/20261016/half-triplet-lower-bound/gpu/verification.json').read_text())
    assert triplet_audit['verified_files']==20 and triplet_audit['precision_diagnostic_outputs']==32 and triplet_audit['synthetic_shape_checks']==6 and triplet_audit['timed_executions']==2400
    assert all(r['gpu_fraction']>2.8 and r['wall_fraction']>1.97 for r in triplet_audit['profiles'] if r['candidate']==1)
    retained_phases=json.loads((p/'retained-control-phase-analysis.json').read_text())
    assert retained_phases['source_archive_sha256']=='1861996b1455f4a6aa7e81fb89bac30a2be2526d359eb332cc974d31aa6e520e'
    assert len(retained_phases['raw_file_sha256'])==18 and not retained_phases['full_goal_met']
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| phase Qwen')]
    assert len(rows)==len(retained_phases['profiles'])==6;retained_phase_numbers=0
    for cells,row in zip(rows,retained_phases['profiles']):
        assert cells[0].strip()=='phase '+row['model'] and int(cells[1])==row['batch'] and row['warm_requests']==4
        intervals=row['wall_intervals_ms'];values=[intervals.get(k) for k in ('ttft_ms','vision_ms','prefill_wall_ms')]+[row['prefill_profile_ms']['gpu_ms']]
        assert [c.strip() for c in cells[2:]]==['—' if value is None else f'{value:.3f}' for value in values]
        retained_phase_numbers+=sum(value is not None for value in values)
    p=ROOT/'results/20261016/half-triplet-lower-bound'
    half_cpu=json.loads((p/'cpu-reconstruction.json').read_text())
    assert half_cpu['helper_source_sha256']==hashlib.sha256((ROOT/'test2026/metal/analyze_half_triplet_reconstruction.py').read_bytes()).hexdigest()
    assert half_cpu['probe_source_sha256']==hashlib.sha256((ROOT/'test2026/metal/mps_half_triplet_lower_bound.mm').read_bytes()).hexdigest()
    assert len(half_cpu['profiles'])==24 and half_cpu['bounded_source_bits_exact'] and not half_cpu['full_goal_met']
    assert sum(r['inputs'] for r in half_cpu['profiles'])==half_cpu['bounded_inputs_checked']==2826240
    assert all(r['admitted']==r['inputs'] and r['requires_original_f32_fallback']==0 for r in half_cpu['profiles'])
    half_edges=half_cpu['finite_exponent_edge_cases'];assert (half_edges['inputs'],half_edges['admitted'],half_edges['requires_original_f32_fallback'])==(4590,769,3821)
    assert len(half_cpu['edge_profiles'])==255 and sum(r['admitted'] for r in half_cpu['edge_profiles'])==769
    p=ROOT/'results/20261015/mlp-prefill-native'
    phase=json.loads((p/'phase-analysis.json').read_text())
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines()
          if line.startswith(('| Text |','| Coder |'))]
    phase_profiles=[v for v in phase['profiles'] if v['model']!='Qwen3.5-0.8B']
    assert len(rows)==len(phase_profiles)==4
    phase_numbers=0
    for cells,r in zip(rows,phase_profiles):
        values=r['policies']['control']['decode_per_step_ms_equal_request_means']
        assert int(cells[1])==r['batch']
        assert [cell.strip() for cell in cells[2:]]==[f'{values[key]:.3f}' for key in ('prepare_ms','encode_ms','gpu_ms','wall_window_per_step_ms')]
        phase_numbers+=4
    union=ROOT/'results/20261016/joined-union-half-admission'
    union_audit=json.loads((union/'verification.json').read_text())
    assert (union_audit['pairs'],union_audit['matrices'],union_audit['union_admitted_pairs'],union_audit['rejected_weight_values'])==(76,152,75,1)
    assert union_audit['captured_original_cpp_roundtrip_values']==561512447
    rows=[line.split('|')[1:-1] for line in (union/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==len(union_audit['models'])==3
    for cells,model in zip(rows,union_audit['models']):
        assert cells[0].strip()==model['model']
        assert [int(v.strip()) for v in cells[1:4]]==[model['pairs'],model['old_equal_scale_pairs'],model['union_pairs']]
    primitive=json.loads((ROOT/'results/20261016/prepared-union-mlp/primitive/verification.json').read_text())
    assert primitive['original_dyadic_cases']==24 and primitive['cache_admitted_pairs']==128 and primitive['cache_rejected_pairs']==2
    assert primitive['regenerated_context_and_header'] and primitive['pending_foreign_alias_abort_recovery_controls'] and not primitive['full_goal_met']
    prepared=ROOT/'results/20261016/prepared-union-mlp'
    decision=json.loads((prepared/'decision-repeat.json').read_text())
    assert not decision['confirmed_models'] and decision['complete_numerical_profiles']==304 and decision['retained_timing_samples']==9600
    cohorts=[json.loads((prepared/label/'decision.json').read_text()) for label in ('initial','repeat')]
    rows=[line.split('|')[1:-1] for line in (prepared/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==12
    for cells,first,second in zip(rows,cohorts[0]['profiles'],cohorts[1]['profiles']):
        assert cells[0].strip()==first['model']==second['model'] and int(cells[1])==first['layer']==second['layer'] and int(cells[2])==first['rows']==second['rows']
        values=[v['fractions']['comparison'][field] for v in (first,second) for field in ('gpu_ms','wall_ms')]
        assert [v.strip() for v in cells[3:]]==[f'{v:.6f}' for v in values]
    costs=json.loads((prepared/'cost-summary.json').read_text())
    assert costs['source_sha256']==hashlib.sha256((ROOT/'test2026/metal/summarize_prepared_union_mlp_costs.py').read_bytes()).hexdigest()
    assert len(costs['cohorts'])==2 and all(sum(v['admitted_layers'] for v in trial['models'])==75 for trial in costs['cohorts'])
    qkv=ROOT/'results/20261016/qkv-union-half-admission';qkv_audit=json.loads((qkv/'verification.json').read_text())
    assert (qkv_audit['groups'],qkv_audit['matrices'],qkv_audit['admitted_groups'],qkv_audit['rejected_values'],qkv_audit['retained_fp32_key_matrices'])==(58,174,56,2,52)
    assert qkv_audit['captured_original_roundtrip_values']==173670398
    rows=[line.split('|')[1:-1] for line in (qkv/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==3
    for cells,model in zip(rows,qkv_audit['models']):
        assert cells[0].strip()==model['model'] and [int(v.strip()) for v in cells[1:]]==[model['separate_groups'],model['admitted_groups'],model['existing_combined_layers']]
    qkv_primitive=json.loads((ROOT/'results/20261016/prepared-union-qkv/primitive/verification.json').read_text())
    assert (qkv_primitive['dyadic_cases'],qkv_primitive['cache_admitted_groups'],qkv_primitive['cache_rejected_groups'],qkv_primitive['initial_extra_union_weight_bytes'])==(24,64,2,336)
    assert qkv_primitive['regenerated_context'] and qkv_primitive['original_fp32_key_fixture'] and not qkv_primitive['full_goal_met']
    prepared_qkv=ROOT/'results/20261016/prepared-union-qkv'
    combined=json.loads((prepared_qkv/'decision-repeat.json').read_text())
    assert combined['confirmed_models']==['Qwen3-0.6B'] and combined['complete_numerical_profiles']==232 and combined['retained_timing_samples']==9600 and not combined['full_goal_met']
    cohorts=[json.loads((prepared_qkv/label/'decision.json').read_text()) for label in ('initial','repeat')]
    rows=[line.split('|')[1:-1] for line in (prepared_qkv/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==12
    for cells,first,second in zip(rows,cohorts[0]['profiles'],cohorts[1]['profiles']):
        assert cells[0].strip()==first['model']==second['model'] and int(cells[1])==first['layer']==second['layer'] and int(cells[2])==first['rows']==second['rows']
        values=[v['fractions']['comparison'][field] for v in (first,second) for field in ('gpu_ms','wall_ms')]
        assert [v.strip() for v in cells[3:]]==[f'{v:.6f}' for v in values]
    costs=json.loads((prepared_qkv/'cost-summary.json').read_text())
    assert costs['source_sha256']==hashlib.sha256((ROOT/'test2026/metal/summarize_prepared_union_qkv_costs.py').read_bytes()).hexdigest()
    assert len(costs['cohorts'])==2 and all(sum(v['admitted_groups'] for v in trial['models'])==56 for trial in costs['cohorts'])
    import runpy,sys
    sys.path.insert(0,str(ROOT/'test2026/metal'))
    factory=runpy.run_path(str(ROOT/'test2026/metal/make_prepared_qkv_native.py'))
    prep=json.loads((ROOT/'results/20261016/prepared-qkv-native/preparation.json').read_text())
    assert len(prep['changes'])==len(factory['OPERATIONS'])==12 and len(prep['added_sources'])==5
    assert prep['selection']['confirmed_models']==['Qwen3-0.6B'] and not prep['full_goal_met']
    for name,operation in factory['OPERATIONS'].items():
        p=ROOT/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==prep['changes'][name]['before']
        assert hashlib.sha256(operation(p.read_text()).encode()).hexdigest()==prep['changes'][name]['after']
    rejected=ROOT/'results/20261016/prepared-qkv-native'
    rejection=json.loads((rejected/'rejection/verification.json').read_text())
    assert rejection['candidate_rejected'] and not rejection['native_performance_admitted'] and not rejection['full_goal_met']
    assert (rejection['total_passing_stages_before_rejection'],rejection['reused_passing_stages'],rejection['fresh_passing_recovery_stages'],rejection['exact_saved_fixture_inputs'],rejection['control_reference_steps'])==(31,25,6,21,3)
    rows=[line.split('|')[1:-1] for line in (rejected/'README.md').read_text().splitlines() if line.startswith(('| Fused Q/K/V','| Accepted d52'))]
    assert len(rows)==2
    assert [row[1].strip() for row in rows]==[f"{rejection[k]:.12f}" for k in ('candidate_failed_original_bound_ratio','accepted_control_worst_original_bound_ratio')]
    inverse=ROOT/'results/20261016/prepared-half-inverse'
    proofs=[json.loads((inverse/label/'verification.json').read_text()) for label in ('windows','macos')]
    assert proofs[0]['counts']==proofs[1]['counts'] and proofs[1]['apple_clang_clz_branch']
    header_sha=hashlib.sha256((ROOT/'test2026/metal/prepared_half_inverse_bits.h').read_bytes()).hexdigest()
    assert all(p['integer_header_sha256']==header_sha and not p['full_goal_met'] for p in proofs)
    counts=proofs[0]['counts'];assert not counts['gpu_or_model_test'] and counts['rejected_outputs_unchanged'] and counts['original_codec_and_float_reference_bits_exact']
    rows=[line.split('|')[1:-1] for line in (inverse/'README.md').read_text().splitlines() if line.startswith(('| Original BF16','| Half words','| Six invalid'))]
    values=[(counts['bf16_scale_cases'],counts['original_codec_admitted'],counts['original_codec_rejected']),(counts['half_scale_cases'],counts['finite_half_exact'],counts['nonfinite_half_rejected']),(counts['invalid_scale_cases'],0,counts['invalid_scale_cases'])]
    assert len(rows)==3 and [[int(v.strip()) for v in row[1:]] for row in rows]==[list(v) for v in values]
    qv=ROOT/'results/20261016/prepared-union-qv';qv_decision=json.loads((qv/'initial/decision.json').read_text());qv_audit=json.loads((qv/'initial/verification.json').read_text())
    assert not qv_decision['unchanged_confirmation_admitted'] and not qv_decision['native_integration_admitted'] and not any(m['component_screen_pass'] for m in qv_decision['models'])
    assert (qv_audit['complete_learned_profiles'],qv_audit['timing_samples'],qv_audit['original_fp32_key_matrices'])==(116,4800,52)
    rows=[line.split('|')[1:-1] for line in (qv/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==15
    for cells,r in zip(rows[:12],qv_decision['profiles']):
        assert cells[0].strip()==r['model'] and int(cells[1])==r['layer'] and int(cells[2])==r['rows']
        values=[r['fractions'][kind][field] for kind in ('comparison','null') for field in ('gpu_ms','wall_ms')]
        assert [v.strip() for v in cells[3:7]]==[f'{v:.6f}' for v in values] and cells[7].strip()==('Yes' if r['profile_pass'] else 'No')
    qv_cost=json.loads((qv/'cost-summary.json').read_text());assert qv_cost['source_sha256']==hashlib.sha256((ROOT/'test2026/metal/summarize_prepared_union_qv_costs.py').read_bytes()).hexdigest()
    costs=qv_cost['cohorts'][0]['models'];assert sum(m['original_key_bit_checks'] for m in costs)==10723328
    for cells,m in zip(rows[12:],costs):
        assert cells[0].strip()==m['model'] and int(cells[1])==m['admitted_groups']
        values=[m['logical_all_admitted_qv_weights_bytes']/2**20,m['first_shape_preparation_ms']['min'],m['first_shape_preparation_ms']['max']]
        assert [v.strip() for v in cells[2:]]==[f'{values[0]:.3f}',f'{values[1]:.6f}',f'{values[2]:.6f}']
    views=ROOT/'results/20261016/prepared-union-qv-views';view_decision=json.loads((views/'initial/decision.json').read_text());view_audit=json.loads((views/'initial/verification.json').read_text())
    assert not view_decision['unchanged_confirmation_admitted'] and not any(m['component_screen_pass'] for m in view_decision['models'])
    assert view_audit['complete_learned_profiles']==116 and view_audit['timing_samples']==4800
    primitive=json.loads((views/'primitive/verification.json').read_text());assert primitive['canonical_view_offsets_and_guard_checked'] and primitive['no_scatter_encode_in_candidate'] and not primitive['downstream_gpu_consumers_adapted']
    rows=[line.split('|')[1:-1] for line in (views/'README.md').read_text().splitlines() if line.startswith('| Qwen')];assert len(rows)==12
    assert sum(r['profile_pass'] for r in view_decision['profiles'])==5
    for cells,r in zip(rows,view_decision['profiles']):
        assert cells[0].strip()==r['model'] and int(cells[1])==r['layer'] and int(cells[2])==r['rows']
        values=[r['fractions'][kind][field] for kind in ('comparison','null') for field in ('gpu_ms','wall_ms')]
        assert [v.strip() for v in cells[3:7]]==[f'{v:.6f}' for v in values] and cells[7].strip()==('Yes' if r['profile_pass'] else 'No')
    paired=ROOT/'results/20261016/paired-norm-rope';pair_audit=json.loads((paired/'initial/verification.json').read_text());pair_analysis=json.loads((paired/'initial/analysis.json').read_text());pair_primitive=json.loads((paired/'primitive/verification.json').read_text())
    assert (pair_audit['complete_learned_profiles'],pair_audit['timing_samples'],pair_audit['summary']['numeric']['original_output_bits_checked'])==(232,9600,47321088)
    assert pair_primitive['original_fp32_output_bits_checked']==3710720 and not pair_analysis['native_integration_admitted']
    assert sum(r['null_within_five_percent'] for r in pair_analysis['profiles'])==8
    rows=[line.split('|')[1:-1] for line in (paired/'README.md').read_text().splitlines() if line.startswith('| Qwen')];assert len(rows)==24
    for cells,r in zip(rows,pair_analysis['profiles']):
        assert cells[0].strip()==r['model'] and int(cells[1])==r['layer'] and int(cells[2])==r['rows']
        values=[r['fractions'][kind][field] for kind in ('comparison','null') for field in ('gpu_ms','wall_ms')]
        assert [v.strip() for v in cells[3:7]]==[f'{v:.6f}' for v in values] and cells[7].strip()==('Yes' if r['null_within_five_percent'] else 'No')
    chain=ROOT/'results/20261016/paired-norm-rope-chain';chain_decision=json.loads((chain/'decision-repeat.json').read_text())
    assert chain_decision['complete_learned_profiles']==464 and chain_decision['timing_samples']==19200 and chain_decision['original_stage_output_bits_checked']==3028549632 and not chain_decision['native_performance_proven']
    assert chain_decision['confirmed_workloads']==[dict(model='Qwen2.5-Coder-0.5B-Instruct',kind='decode'),dict(model='Qwen3-0.6B',kind='decode')]
    cohorts=[json.loads((chain/label/'analysis.json').read_text()) for label in ('initial','repeat')]
    assert [sum(v['null_within_five_percent'] for v in c['profiles']) for c in cohorts]==[24,21]
    rows=[line.split('|')[1:-1] for line in (chain/'README.md').read_text().splitlines() if line.startswith('| Qwen')];assert len(rows)==30
    for cells,w in zip(rows[:6],chain_decision['workloads']):
        assert cells[0].strip()==w['model'] and cells[1].strip()==w['kind'] and [v.strip() for v in cells[2:]]==['Pass' if w['initial_pass'] else 'Fail','Pass' if w['repeat_pass'] else 'Fail','Yes' if w['confirmed'] else 'No']
    for cells,a,b in zip(rows[6:],cohorts[0]['profiles'],cohorts[1]['profiles']):
        assert cells[0].strip()==a['model']==b['model'] and int(cells[1])==a['layer']==b['layer'] and int(cells[2])==a['rows']==b['rows']
        values=[v['fractions']['comparison'][field] for v in (a,b) for field in ('gpu_ms','wall_ms')]
        assert [v.strip() for v in cells[3:7]]==[f'{v:.6f}' for v in values] and [v.strip() for v in cells[7:]]==['Yes' if v['profile_screen_pass'] else 'No' for v in (a,b)]
    paired_native=ROOT/'results/20261016/paired-norm-rope-native';paired_prep=json.loads((paired_native/'preparation.json').read_text());paired_factory=runpy.run_path(str(ROOT/'test2026/metal/make_paired_norm_rope_native.py'));paired_ops=paired_factory['operations'](paired_prep['selected_models'])
    assert len(paired_ops)==len(paired_prep['changes'])==12 and len(paired_prep['added_sources'])==5 and paired_prep['selection']==chain_decision
    for name,operation in paired_ops.items():
        p=ROOT/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==paired_prep['changes'][name]['before'] and hashlib.sha256(operation(p.read_text()).encode()).hexdigest()==paired_prep['changes'][name]['after']
    for name,sha in paired_prep['added_sources'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==sha
    native_audit=json.loads((paired_native/'checks/verification.json').read_text());assert native_audit==json.loads((paired_native/'checks/mac-verification.json').read_text())
    assert native_audit['validation_stages']==34 and native_audit['paired_rope_trained_steps']==48 and not native_audit['reused_validation_stages'] and not native_audit['reused_compile_stages']
    assert native_audit['paired_rope_serving_route_checks']==36 and native_audit['paired_rope_fallback_route_checks']==12 and native_audit['paired_rope_primitive_output_bits']==3710720
    gate=json.loads((paired_native/'checks/gate.json').read_text());assert gate['archive_sha256']==json.loads((paired_native/'checks/archive-verification.json').read_text())['archive_sha256'] and gate['candidate_sha256']==native_audit['candidate_library_sha256']
    assert hashlib.sha256((paired_native/'checks/mac-verification.json').read_bytes()).hexdigest()==gate['full_audit_sha256'] and not gate['full_goal_met']
    tiles=ROOT/'results/20261016/delta-value-tiles-addresses';tile_audit=json.loads((tiles/'verified/verification.json').read_text());tile_report=json.loads((tiles/'verified/evidence/report.json').read_text())
    assert (tile_audit['reproduced_cpu_profiles'],tile_audit['state_words_checked'],tile_audit['output_words_checked'])==(12,4718592,3575808)
    assert not any(tile_audit[k] for k in ('gpu_executed','numerical_equivalence_proven','performance_measured','full_goal_met'))
    assert {(p['width'],p['batch'],p['padding_floats']) for p in tile_report['profiles']}=={(w,b,p) for w in (16,32,64) for b in (1,2) for p in (0,4)}
    provenance=json.loads((tiles/'source-provenance.json').read_text());assert provenance['recurrence_unchanged_by_paired_native_integration'] and provenance['failed_and_fixed_cpu_audits_generated_identical_shaders']
    assert (tiles/'failure/evidence/delta-value-tiles.metal').read_bytes()==(tiles/'verified/evidence/delta-value-tiles.metal').read_bytes()
    native_decision=json.loads((paired_native/'decision-initial.json').read_text());native_profiles=native_decision['trials'][0]['profiles'];assert not native_decision['repeat_admitted'] and not native_decision['native_confirmation_pass'] and not native_decision['promote']
    assert native_decision==json.loads((paired_native/'continuation/evidence/validation-audits/paired-norm-rope-native-results/decision-initial.json').read_text())
    assert json.loads((paired_native/'initial/verification.json').read_text())['full_output_requests']==72
    rows=[line.split('|')[1:-1] for line in (paired_native/'README.md').read_text().splitlines() if line.startswith('| Qwen')];assert len(rows)==12
    for cells,r in zip(rows[:6],native_profiles):
        assert cells[0].strip()==r['model'] and int(cells[1])==r['batch'];a,b=r['means']['control'],r['means']['candidate']
        assert cells[2].strip()==f"{a['decode_tps']:.4f} / {b['decode_tps']:.4f}" and cells[3].strip()==f"{a['ttft_ms']:.4f} / {b['ttft_ms']:.4f}" and [v.strip() for v in cells[4:]]==[f"{r['decode_ratio']:.6f}",f"{r['ttft_fraction']:.6f}"]
    for cells,r in zip(rows[6:],native_profiles):
        assert cells[0].strip()==r['model'] and int(cells[1])==r['batch'];a,b=r['first_request_after_initialization']['control'],r['first_request_after_initialization']['candidate']
        assert cells[2].strip()==f"{a['initialization_s']:.6f} / {b['initialization_s']:.6f}" and cells[3].strip()==f"{a['ttft_ms']:.4f} / {b['ttft_ms']:.4f}" and cells[4].strip()==f"{r['added_mean_worker_peak_rss_gib']*1024:.4f}"
    tile_gpu=ROOT/'results/20261016/delta-value-tiles-gpu';tile_decision=json.loads((tile_gpu/'decision-repeat.json').read_text());assert tile_decision==json.loads((tile_gpu/'mac-decision-repeat.json').read_text())
    assert tile_decision['confirmed_workloads']==[dict(width=64,kind='prefill')] and tile_decision['complete_numeric_profiles']==416 and tile_decision['original_word_comparisons']==1107973296 and tile_decision['timing_samples']==12800 and not tile_decision['native_performance_proven'] and not tile_decision['target_count_update']
    tile_cohorts=[]
    for label in ('initial','repeat'):
        audit=json.loads((tile_gpu/label/'verification.json').read_text());analysis=json.loads((tile_gpu/label/'analysis.json').read_text())
        assert audit==json.loads((tile_gpu/label/'mac-verification.json').read_text()) and analysis==json.loads((tile_gpu/label/'mac-analysis.json').read_text())
        assert audit['numeric_checks']==1872 and audit['original_word_comparisons']==553986648 and audit['timing_samples']==6400 and len(analysis['profiles'])==16 and all(p['own_null_stable'] for p in analysis['profiles']);tile_cohorts.append(analysis)
    rows=[line.split('|')[1:-1] for line in (tile_gpu/'README.md').read_text().splitlines() if re.match(r'^\| (32|64) \|',line)];assert len(rows)==16
    for cells,a,b in zip(rows,tile_cohorts[0]['profiles'],tile_cohorts[1]['profiles']):
        assert [int(c.strip()) for c in cells[:4]]==[a['width'],a['layer'],a['batch'],a['chunk']]==[b['width'],b['layer'],b['batch'],b['chunk']]
        for cell,key in zip(cells[4:6],('gpu_ms','wall_ms')):assert cell.strip()==' / '.join(f"{p['fractions']['comparison'][key]:.6f}" for p in (a,b))
        assert cells[6].strip()==' / '.join('Yes' if p['own_null_stable'] else 'No' for p in (a,b)) and cells[7].strip()==' / '.join('Yes' if p['component_screen_pass'] else 'No' for p in (a,b))
    record = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  reports_verified=len(expected), local_links_checked=links,
                  verification_hashes_checked=verified, matched_table_values_checked=numbers,
                  component_table_values_checked=component_numbers,native_table_values_checked=native_numbers,
                  selected_native_table_values_checked=selected_native_numbers,bf16_component_table_values_checked=bf16_numbers,
                  phase_table_values_checked=phase_numbers,null_calibration_table_values_checked=null_numbers,ndarray_component_table_values_checked=ndarray_numbers,
                  ndarray_native_table_values_checked=ndarray_native_numbers,weight_admission_table_values_checked=admission_numbers,
                  ndarray_selective_component_table_values_checked=selective_numbers,compact_escape_cpu_table_values_checked=compact_numbers,
                  retained_control_guard_preflight_requests=36,selective_native_table_values_checked=selective_native_numbers,
                  vector_initial_table_values_checked=vector_initial_numbers,vector_null_calibration_table_values_checked=vector_calibration_numbers,
                  vector_calibrated_component_table_values_checked=vector_calibrated_numbers,
                  retained_control_phase_table_values_checked=retained_phase_numbers,
                  half_triplet_cpu_profiles_checked=24,half_triplet_finite_edge_cases_checked=4590,
                  vector_native_initial_table_values_checked=vector_native_numbers,
                  vector_native_repeat_table_values_checked=vector_repeat_numbers,validated_branch_source_files=8,
                  compact_group_checks_audited=16,synthetic_triplet_shape_checks_audited=6,
                  union_learned_pairs_cpu_audited=76,prepared_union_primitive_cases_audited=24,
                  prepared_union_learned_table_values_checked=48,prepared_union_complete_numerical_profiles=304,
                  qkv_union_cpu_groups_audited=58,qkv_union_cpu_table_values_checked=9,
                  prepared_qkv_primitive_cases_audited=24,prepared_qkv_learned_table_values_checked=48,
                  prepared_qkv_complete_numerical_profiles=232,prepared_qkv_private_changes_regenerated=12,
                  rejected_qkv_table_values_checked=2,half_inverse_cpu_table_values_checked=9,
                  half_inverse_actual_compiler_hosts=2,qv_rejected_component_table_values_checked=48,
                  qv_preparation_cost_table_values_checked=9,qv_output_view_table_values_checked=48,
                  paired_norm_rope_table_values_checked=96,paired_norm_rope_learned_profiles=232,
                  paired_chain_table_values_checked=96,paired_chain_learned_profiles=464,
                  paired_native_private_transformations_regenerated=12,paired_native_full_validation_stages=34,
                  paired_native_rejected_model_table_values_checked=66,
                  delta_value_tile_cpu_profiles_checked=12,status='passed',
                  delta_value_tile_gpu_endpoint_table_values_checked=64,delta_value_tile_gpu_numeric_profiles=416,
                  scope='Documentation consistency only; experiment numerical/performance audits are separate records')
    (OUT / 'documentation-check.json').write_bytes((json.dumps(record, indent=2)+'\n').encode())
    print(json.dumps(record))


if __name__ == '__main__':
    main()
