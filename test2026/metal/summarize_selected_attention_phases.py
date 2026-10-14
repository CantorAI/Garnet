"""Summarize existing native phase telemetry without rerunning inference.

GPU and wait intervals overlap: they must not be added. Phase observations
are not serialized per-operation timings and do not identify a kernel cause.
"""
import json, statistics, sys
from pathlib import Path

root = Path(sys.argv[1]).resolve()
job = root / 'evidence/logs/selected-attention-vllm-20261013/jobs'
profiles = []
for folder in sorted(job.glob('*-b*-0')):
    summary = json.loads((folder / 'summary.json').read_text())
    records = []
    for log in sorted(folder.glob('garnet-*/output.log')):
        for line in log.read_text().splitlines():
            if line.startswith(('matched_result ', 'matched_frame_result ')):
                records.append(json.loads(line.split(' ', 1)[1]))
    warm = [r for r in records if r['request'] > 0]
    assert len(warm) == 4, (folder.name, len(warm))
    def mean(key):
        return statistics.mean(r[key] for r in warm)
    result = dict(profile=folder.name, warm_requests=len(warm),
                  ttft_ms=mean('ttft_seconds') * 1000)
    for key in ('prefill_inference_ms', 'vision_seconds', 'assembly_seconds',
                'prefill_wall_seconds', 'common_preprocessing_seconds'):
        if all(key in r for r in warm):
            result[key] = mean(key)
    for phase_name in ('prefill_profile', 'decode_phases'):
        if all(phase_name in r for r in warm):
            phases = [p for r in warm for p in
                      ([r[phase_name]] if isinstance(r[phase_name], dict)
                       else r[phase_name])]
            result[phase_name] = {
                key: statistics.mean(p[key] for p in phases)
                for key in phases[0] if key.endswith('_ms')}
    profiles.append(result)
destination = root / 'native-phases.json'
destination.write_bytes((json.dumps(dict(
    scope='Existing warm telemetry; GPU/wait overlap; no kernel attribution',
    profiles=profiles), indent=2) + '\n').encode())
print(destination)
