"""Generate and execute a separately identified carried-chain calibration; keep one-pair evidence frozen."""
from pathlib import Path
import ast,sys
def expand(source):
 def replace(old,new,count=1):
  nonlocal source
  assert source.count(old)==count,(old,source.count(old),count);source=source.replace(old,new)
 replace('from make_paired_norm_rope import context,header','from make_paired_norm_rope import context,header\nfrom make_paired_norm_rope_chain import probe')
 replace("generated=out/'generated';generated.mkdir();", "generated=out/'generated';generated.mkdir();cpp=generated/'paired_norm_rope_chain.mm';cpp.write_bytes(probe((s/'paired_norm_rope_learned.mm').read_text()).encode());")
 replace("s/'paired_norm_rope_learned.mm',", "s/'paired_norm_rope_learned.mm',s/'make_paired_norm_rope_chain.py',s/'run_paired_norm_rope_learned.py',")
 replace("s/'launch_paired_norm_rope_learned.py'", "s/'launch_paired_norm_rope_chain.py'")
 replace("timing_design=dict(common_warmup_ms=2000", "timing_design=dict(chain_stages=32,carried_outputs=True,independent_outputs_per_stage=True,all_stage_outputs_checked=True,common_warmup_ms=2000")
 replace("limitations='Learned native", "limitations='Distinct 32-stage carried-chain calibration: no unchanged confirmation of one-pair data. Learned native")
 replace("binary=out/'paired_norm_rope_learned_probe'", "binary=out/'paired_norm_rope_chain_probe'")
 replace("str(s/'paired_norm_rope_learned.mm')", "str(cpp)")
 ast.parse(source);return source
if __name__=='__main__':
 original=Path(__file__).with_name('run_paired_norm_rope_learned.py');source=expand(original.read_text());destination=Path(sys.argv[2]).with_suffix('.chain.executed.py');destination.write_bytes(source.encode())
 if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
