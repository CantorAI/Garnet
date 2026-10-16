import sys,json,math
sys.path.insert(0,sys.argv[1]);sys.path.insert(0,sys.argv[2])
import garnet
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
folder=sys.argv[3]
with open(folder+'/config.json') as stream:config=json.load(stream)['vision_config']
encoder=MetalImageEncoder(folder+'/model.safetensors',config,2,2,1)
try:
 result=encoder.graph.run([garnet.tensor_io().read_f32(sys.argv[4],[1,4,1536])]).tolist()
 bad=sum([0 if math.isfinite(x) else 1 for x in result])
 print('stage',sys.argv[5],'count',len(result),'nonfinite',bad,'first',result[:8],flush=True)
finally:encoder.close()
