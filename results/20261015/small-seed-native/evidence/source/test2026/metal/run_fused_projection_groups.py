"""One GPU dispatch for independent unchanged BF16 dot products; component study."""
from pathlib import Path
from make_fused_projection_group import generate
source=Path(__file__).with_name('run_concurrent_bf16_groups.py').read_text()
source=source.replace("paths=[native,s/'concurrent_bf16_group_gpu.mm',Path(__file__),s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py']", "paths=[native,s/'concurrent_bf16_group_gpu.mm',s/'run_concurrent_bf16_groups.py',s/'make_fused_projection_group.py',Path(__file__),s/'run_constant_bf16_probe.py',s/'compare_vllm_metal.py']")
source=source.replace("'1':'One serial encoder','2':'One concurrent encoder, independent writes'", "'1':'One dispatch with disjoint projection grid ranges, original dot products'")
start=source.index("text=native.read_text();shader=out/'native-decode.metal'")
end=source.index("metadata['binary_sha256']=digest(binary)",start)
source=source[:start]+'''shader=out/'fused-projections.metal';cpp=out/'fused_projection_group_gpu.mm'
shader_text,cpp_text=generate(native.read_text(),(s/'concurrent_bf16_group_gpu.mm').read_text())
shader.write_bytes(shader_text.encode());cpp.write_bytes(cpp_text.encode())
metadata['native_shader_sha256']=digest(shader);metadata['generated_harness_sha256']=digest(cpp)
binary=out/'fused_projection_group_gpu'
run('compile',['/usr/bin/xcrun','clang++','-std=c++17','-fobjc-arc','-O3','-framework','Foundation','-framework','Metal',str(cpp),'-o',str(binary)])
'''+source[end:]
source=source.replace('for mode in (1,2):','for mode in (1,):').replace("c['validated_executions']>=252","c['validated_executions']>=136")
destination=Path(__import__('sys').argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
