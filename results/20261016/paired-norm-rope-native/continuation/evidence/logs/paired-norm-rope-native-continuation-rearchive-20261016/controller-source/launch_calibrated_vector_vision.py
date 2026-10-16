"""Prospective calibration/candidate probe; original chain and separate null gate enforced."""
from pathlib import Path
import json,sys
root=Path(sys.argv[1]).resolve();mode=sys.argv[3];assert mode in ('calibration','candidate')
if mode=='candidate':
    decision=json.loads((root/'validation-audits/vector-vision-calibrated/calibration-decision.json').read_text())
    assert decision['candidate_probe_admitted'] and len(decision['trials'])==2
source=Path(__file__).with_name('launch_vector_vision_context.py').read_text()
source=source.replace("mode in ('component',)","mode in ('calibration','candidate')")
source=source.replace("runner='run_vector_vision_context.py'","runner='run_calibrated_vector_vision_context.py'")
needle="exec(compile(source,__file__,'exec'))";assert source.count(needle)==1
addition='source=source.replace('+repr("str(out/'jobs')])")+','+repr("str(out/'jobs'),*(['--null-only'] if mode=='calibration' else [])])")+')\n'
source=source.replace(needle,addition+needle)
exec(compile(source,__file__,'exec'))
