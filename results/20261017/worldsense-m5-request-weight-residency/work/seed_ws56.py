from pathlib import Path
import json,hashlib,shutil
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=root/'work/worldsense-request-weight-residency';s=r/'out/research/worldsense-request-weight-residency/source';seed=r/'out/research/worldsense-isolated-primary/source';m=json.loads((root/'work/worldsense-isolated-primary/correctness-source-manifest.json').read_text());assert json.loads((w/'protocol.json').read_text())['experiment']=='WS-M5-56'and not s.exists();assert len(m)==1718 and all(hashlib.sha256((seed/k).read_bytes()).hexdigest()==h for k,h in m.items());s.mkdir(parents=True)
for rel in m:(s/rel).parent.mkdir(parents=True,exist_ok=True);shutil.copy2(seed/rel,s/rel)
(w/'ws54-seed-source-manifest.json').write_text(json.dumps(m,indent=2)+'\n')
budget=r/'out/research/worldsense-weight-residency/source/src/backends/metal/weight_residency_budget.h';shutil.copy2(budget,s/'src/backends/metal/weight_residency_budget.h');(w/'budget-source-identity.json').write_text(json.dumps({'source':str(budget),'sha256':hashlib.sha256(budget.read_bytes()).hexdigest(),'unchanged_WS32_budget_implementation':True},indent=2)+'\n')
print('WS56 seed verified/copied; unchanged bounded-budget helper imported')
