"""All 12 pretrained vision blocks and merger against a FP32 reference."""
import sys,json,struct,tempfile,subprocess
from pathlib import Path
import numpy as np
from reference_metal_vision import reference
folder=Path(sys.argv[4]);checkpoint=folder/'model.safetensors'
config=json.loads((folder/'config.json').read_text())['vision_config']
with checkpoint.open('rb') as stream:
    size=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(size))
weights={}
for name,entry in header.items():
    if not name.startswith('model.visual.'):continue
    if entry['dtype']!='BF16':raise ValueError('unexpected vision tensor dtype')
    encoded=np.memmap(checkpoint,mode='r',dtype='<u2',offset=8+size+entry['data_offsets'][0],shape=tuple(entry['shape']))
    weights[name[len('model.visual.'):]]=(encoded.astype(np.uint32)<<16).view(np.float32)
rng=np.random.default_rng(768)
for batch in (1,2):
    # Small grid bounds validation cost; real image profile runs separately.
    patches=rng.normal(size=(batch,4,1536)).astype(np.float32)
    expected=reference(patches,weights,config,2,2)
    with tempfile.TemporaryDirectory(prefix='garnet-real-vision-') as temporary:
        patches_path=Path(temporary)/'patches.f32';patches.astype('<f4').tofile(patches_path)
        expected_path=Path(temporary)/'expected.f32';expected.astype('<f4').tofile(expected_path)
        fixture=dict(config=config,checkpoint=str(checkpoint.resolve()),shape=list(patches.shape),grid_h=2,grid_w=2,
            steps=[dict(patch_file=str(patches_path),expected_file=str(expected_path),expected_count=expected.size)])
        path=Path(temporary)/'fixture.json';path.write_text(json.dumps(fixture))
        subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_vision_graph.py')),str(path),sys.argv[2],sys.argv[3]],check=True)
print('all pretrained vision blocks/merger batch 1/2 reference PASS',flush=True)

