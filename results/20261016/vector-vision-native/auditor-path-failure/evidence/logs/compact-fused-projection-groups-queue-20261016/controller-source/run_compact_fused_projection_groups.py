"""Compact-aware grouped decode plus same-pipeline null; no full-model gain claim."""
from pathlib import Path
import sys
source=Path(__file__).with_name('run_concurrent_bf16_groups.py').read_text()
source=source.replace('from pathlib import Path','from pathlib import Path\nfrom make_compact_fused_projection_groups import generate',1)
old="paths=[native,s/'concurrent_bf16_group_gpu.mm',Path(__file__),s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py']"
assert source.count(old)==1
source=source.replace(old,"paths=[native,root/'source/src/backends/metal/coalesced_body_block_shader.h',s/'concurrent_bf16_group_gpu.mm',s/'make_compact_fused_projection_groups.py',s/'run_concurrent_bf16_groups.py',Path(__file__),s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py',s/'verify_concurrent_bf16_groups.py',s/'verify_compact_fused_projection_groups.py']")
source=source.replace("'0':'Separate serial encoders','1':'One serial encoder','2':'One concurrent encoder, independent writes'","'0':'Retained compact64/raw128/small64 separate encoders','1':'Compact-aware grouped64, separate outputs','2':'Same retained pipelines and geometry null'")
start=source.index("text=native.read_text();shader=out/'native-decode.metal'");end=source.index("metadata['binary_sha256']=digest(binary)",start)
source=source[:start]+'''shader=out/'compact-fused-projections.metal';cpp=out/'compact_fused_projection_group_gpu.mm'
shader_text,cpp_text=generate(native.read_text(),(root/'source/src/backends/metal/coalesced_body_block_shader.h').read_text(),(s/'concurrent_bf16_group_gpu.mm').read_text())
shader.write_bytes(shader_text.encode());cpp.write_bytes(cpp_text.encode())
metadata['native_shader_sha256']=digest(shader);metadata['generated_harness_sha256']=digest(cpp)
metadata['timing_design']=dict(common_warmup_ms=2000,rounds=10,window_order='ABBA',samples_per_window=5,warm_calls_per_window=3,all_samples_retained=True)
binary=out/'compact_fused_projection_group_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(cpp),'-o',str(binary)])
'''+source[end:]
needle="grouped=dict(groups=groups,checkpoint_sha256=info['checkpoint_sha256'])";assert source.count(needle)==1
source=source.replace(needle,"""for group in groups:
        for case in group['cases']:
            case['float_key']=case['tensor'].startswith('model.layers.') and case['tensor'].endswith('.self_attn.k_proj.weight')
    """+needle)
source=source.replace("c['validated_executions']>=252","c['validated_executions']>=652").replace('assert len(windows)==8','assert len(windows)==40').replace('for rnd in (0,1):','for rnd in range(10):').replace('len(v[k])==12','len(v[k])==5')
source=repr(__doc__)+source[source.index('\n'):]
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
