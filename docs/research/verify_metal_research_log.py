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
               'coalesced-eight-body','mlp-command-buffer-primitive','mlp-command-buffer-native','eight-fma-body')]
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
    # Rejected full-native integration: preserve its slower timings and memory.
    p=ROOT/'results/20261015/mlp-prefill-native'
    native=json.loads((p/'comparison/verification.json').read_text())['profiles']
    rows=[line.split('|')[1:-1] for line in (p/'README.md').read_text().splitlines() if line.startswith('| Qwen')]
    assert len(rows)==len(native)==6
    native_numbers=0
    for cells,r in zip(rows,native):
        assert int(cells[1])==r['batch']
        means=r['means'];first=r['first_request_after_initialization'];rss=r['peak_individual_rss_bytes']
        formatted=[
            ' / '.join(f"{means[k]['ttft_ms']:.3f}" for k in ('control','candidate')),
            f"{r['decode_ratio']:.3f}",
            ' / '.join(f"{first[k]['ttft_ms']:.3f}" for k in ('control','candidate')),
            f"{(rss['candidate']-rss['control'])/2**30:.3f}"]
        assert [c.strip() for c in cells[2:]]==formatted
        native_numbers+=6
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
    record = dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
                  reports_verified=len(expected), local_links_checked=links,
                  verification_hashes_checked=verified, matched_table_values_checked=numbers,
                  component_table_values_checked=component_numbers,native_table_values_checked=native_numbers,phase_table_values_checked=phase_numbers,status='passed',
                  scope='Documentation consistency only; experiment numerical/performance audits are separate records')
    (OUT / 'documentation-check.json').write_bytes((json.dumps(record, indent=2)+'\n').encode())
    print(json.dumps(record))


if __name__ == '__main__':
    main()
