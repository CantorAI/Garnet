"""Audit the specified repeat profiles with identical complete-output gates."""
from pathlib import Path
source=Path(__file__).with_name('verify_packed_prefill_norm_native.py').read_text()
source=source.replace("folder=base/'logs/packed-prefill-norm-resume-native-20261014/jobs'","folder=base/'logs/packed-prefill-norm-confirm-native-20261014/jobs'")
source=source.replace("{(m,b) for m in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B') for b in (1,2)}","{('Qwen3-0.6B',2),('Qwen2.5-Coder-0.5B-Instruct',1),('Qwen3.5-0.8B',2)}")
needle="assert proof['jobs']['packed-prefill-norm-checks-20261014']['exit_code']"
assert source.count(needle)==1
source=source.replace(needle,"assert proof['jobs']['packed-prefill-norm-confirm-native-20261014']['exit_code']==0\n"+needle)
exec(compile(source,str(Path(__file__)),'exec'))
