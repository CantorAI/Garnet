"""Original64 plus64 chained cases using BF16 feed storage and FP32 arithmetic."""
from pathlib import Path
import sys
p=Path(__file__).with_name('run_mlp_command_buffer_primitive.py');source=p.read_text()
old='from make_mlp_command_buffer_context import context,graph,primitive';assert source.count(old)==1
source=source.replace(old,'from make_mlp_bf16_feed_graph import context,graph,primitive')
old="s/'make_mlp_command_buffer_context.py'";assert source.count(old)==1
source=source.replace(old,"s/'make_mlp_bf16_feed_graph.py',s/'run_mlp_command_buffer_primitive.py',"+old)
old="chained=out.parent.name=='mlp-command-buffer-chained-20261015'";assert source.count(old)==1
source=source.replace(old,'chained=True')
old="    'MPSCommandBuffer.h':'MetalPerformanceShaders.framework/Frameworks/MPSCore.framework/Headers/MPSCommandBuffer.h'}"
new="    'MPSCommandBuffer.h':'MetalPerformanceShaders.framework/Frameworks/MPSCore.framework/Headers/MPSCommandBuffer.h',\n    'MPSCoreTypes.h':'MetalPerformanceShaders.framework/Versions/A/Frameworks/MPSCore.framework/Versions/A/Headers/MPSCoreTypes.h'}"
assert source.count(old)==1;source=source.replace(old,new)
source=source.replace("('MPSGraph.h','MPSCommandBuffer.h')","('MPSGraph.h','MPSCommandBuffer.h','MPSCoreTypes.h')")
source=source.replace("scope=__doc__,source_sha256", "scope=__doc__,weight_storage='original BF16 buffers',arithmetic='FP32 after exact graph cast; no activation precision change',source_sha256")
source=source.replace("binary=out/'mlp_command_buffer_gpu'","binary=out/'mlp_bf16_feed_gpu'")
source=repr(__doc__)+source[source.index('\n'):]
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,__file__,'exec'))
