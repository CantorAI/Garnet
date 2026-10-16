"""Sequential attention-prefix component references and ABBA timing."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_vision_fused_context_probe.py').read_text()
source=source.replace('vision_fused_context','attention_prefix')
source=source.replace('FP32 vision softmax and value contraction component only; synthetic inputs, no full model claim','FP32 chunk attention verified-prefix component only; synthetic inputs, general-scan fallback, no full model claim')
source=source.replace('assert len(checks)==24','assert len(checks)==18')
source=source.replace("('batch','tokens','heads','dim')","('batch','chunk','heads','kv_heads','dim')")
source=source.replace("r['policy']=='fused'","r['policy']=='prefix'")
source=source.replace('fused_gpu_ms=candidate','prefix_gpu_ms=candidate')
source=source.replace("r['tokens']==216 and r['heads']==12 and r['dim']==64","r['heads']==8 and r['dim']==256")
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
