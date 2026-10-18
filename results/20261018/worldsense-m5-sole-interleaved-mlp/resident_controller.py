"""Sequential correctness-only graph/session interoperability matrix."""
from pathlib import Path
import subprocess,json
w=Path(__file__).resolve().parent;n='/Users/shawn/CantorAI/out/build/macos-arm64-release/bin/xlang3';source='/Users/shawn/CantorAI/out/research/worldsense-sole-interleaved/source';results=[]
for mode in ['off','primary','on']:
 for fixture in ['fixture-1-8','fixture-2-64']:
  log=w/f'resident-{mode}-{fixture}.log'
  with log.open('w')as stream:
   p=subprocess.run([n,'-B',str(w/'resident_guard.py'),str(w/fixture/'fixture.json'),'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-sole-interleaved-'+mode,source],stdout=stream,stderr=subprocess.STDOUT)
  results.append(dict(mode=mode,fixture=fixture,returncode=p.returncode,log=log.name));(w/'resident-results.json').write_text(json.dumps(results,indent=2))
  if p.returncode:raise RuntimeError(str(results[-1]))
print('OFF/ON ordinary-resident alternation, late second session, invalid inputs, closed/rebuilt graphs PASS',flush=True)
