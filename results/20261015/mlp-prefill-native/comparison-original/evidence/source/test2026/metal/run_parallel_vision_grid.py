"""Observe trained full-grid embeddings, dispatch counts, fresh encoding and close."""
import sys,json
sys.path.insert(0,sys.argv[2]);sys.path.insert(0,sys.argv[3])
import garnet
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder
with open(sys.argv[1]) as stream:f=json.load(stream)
encoder=MetalImageEncoder(f['checkpoint'],f['config'],f['grid_h'],f['grid_w'],f['batch'])
enabled='parallel_vision_context' in garnet.backend_info();results=[];counts=[]
try:
    for repeat in range(2):
        results.append(encoder.encode_file(f['patch_file']).tolist())
        counts.append(encoder.graph.vision_parallel_dispatches if enabled else 0)
    assert results[0]==results[1],'Fresh trained encoder repeat differs'
    assert counts==([f['config']['depth']]*2 if enabled else [0,0])
    with open(f['output_file'],'w') as stream:json.dump(dict(values=results[0],counts=counts,enabled=enabled),stream)
    encoder.close()
    if enabled:assert encoder.graph.vision_parallel_dispatches==0,'Closed vision counter retained'
    print('trained full-grid fresh encoding and counters PASS',f['batch'],enabled,counts,flush=True)
finally:
    encoder.close();garnet.shutdown()
