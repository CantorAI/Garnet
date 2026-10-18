"""Archive exact-source configuration experiment and unchanged full raw records."""
from pathlib import Path
import ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt'
t=(f/'archive_ws70.py').read_text().replace('WS70','WS71').replace('WS-M5-70','WS-M5-71').replace('worldsense-primary-core-isolation','worldsense-minimal-full-prefill').replace('worldsense-m5-primary-core-isolation','worldsense-m5-minimal-full-prefill').replace('ws70','ws71')
t=t.replace('Valid rejection: both model TTFT,4Btotal andwholeSkillwall fail unchanged2%gates, despite favorable observed deriveddecode. No later stages/promotion.','Valid rejection: textTTFT and4Bderiveddecode fail unchanged2%gates despite favorable observed4BTTFT/total/wholeSkill. No later stages/promotion.')
needle="for p in sorted(Path(__file__).parent.glob('*ws71*.py')):copy(p,d/'work'/p.name)"
extra="""for name in load(w/'learned-guard-adaptation-proof.json'):
 copy(w.parent/'worldsense-primary-core-isolation'/name,d/'frozen-parent-methods'/name)
"""
assert t.count(needle)==1;t=t.replace(needle,needle+'\n'+extra)
needle=" elif p.is_relative_to(r/'out/research/worldsense-minimal-full-prefill/source'):"
extra=""" elif p.parent==w.parent/'worldsense-primary-core-isolation'and(d/'frozen-parent-methods'/p.name).is_file():
  target=d/'frozen-parent-methods'/p.name;assert sha(target)==digest;bundled=str(target.relative_to(d))
"""
assert t.count(needle)==1;t=t.replace(needle,extra+needle)
ast.parse(t);p=f/'archive_ws71.py';assert not p.exists();p.write_text(t)
print('WS71 archiver ready: exact1721source/16canonical overlays, full original evidence/failed preparation attempts,32calls16requests; no copies or inference yet.')
