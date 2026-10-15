"""Repeat two TTFT signals and the text-B2 GPU/total-latency discrepancy."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_packed_prefill_norm_profiles.py').read_text()
source=source.replace("assert '--confirm' not in sys.argv[3:]","assert '--confirm' in sys.argv[3:]")
needle='destination=Path(sys.argv[2]).with_suffix';assert source.count(needle)==1
source=source.replace(needle,"source=source.replace(\"[('Qwen3-0.6B',2),('Qwen2.5-Coder-0.5B-Instruct',1)]\",\"[('Qwen3-0.6B',2),('Qwen2.5-Coder-0.5B-Instruct',1),('Qwen3.5-0.8B',2)]\").replace('assert len(profiles)==6','assert len(profiles)==3')\n"+needle)
exec(compile(source,str(Path(__file__)),'exec'))
