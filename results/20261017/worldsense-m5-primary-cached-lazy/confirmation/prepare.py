from pathlib import Path
import json,shutil
w=Path(__file__).resolve().parent;p=w.parent
assert json.loads((p/'speed/analysis.json').read_text())['passed']
assert json.loads((p/'primary-speed/analysis.json').read_text())['passed']
assert json.loads((p/'production-speed/analysis.json').read_text())['passed']
assert json.loads((p/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
assert not list(w.glob('control-full-*.json')) and not list(w.glob('candidate-full-*.json'))
c=json.loads((p.parent/'worldsense-vision-workspace/confirmation/cases.json').read_text());assert c['repeats']==1 and not c['paired_prefill'];(w/'cases.json').write_text(json.dumps(c,indent=2))
s=(p/'speed/controller.py').read_text().replace('work/worldsense-primary-cached-lazy/speed','work/worldsense-primary-cached-lazy/confirmation')
(w/'controller.py').write_text(s)
(w/'execution-plan.json').write_text(json.dumps(dict(status='prepared_after_speed_pass',order=['control-full-1','candidate-full-1','candidate-full-2','control-full-2'],each_pair_passes_independently=True,full_calls_per_worker=12),indent=2))
