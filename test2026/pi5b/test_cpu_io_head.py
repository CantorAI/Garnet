"""Independent embedding, native row splitting and tied-head reference."""
import sys,json,tempfile,subprocess
from pathlib import Path
import numpy as np
from checkpoint_fixture import write_checkpoint
rng=np.random.default_rng(87)
with tempfile.TemporaryDirectory(prefix='garnet-cpu-head-') as folder:
    folder=Path(folder);checkpoint=folder/'weights.safetensors'
    weights=write_checkpoint(checkpoint,dict(embedding=rng.normal(size=(19,12)).astype(np.float32),
        norm=rng.normal(size=12).astype(np.float32)),dict(embedding='embed_tokens.weight',norm='norm.weight'),'model.language_model')
    x=rng.normal(size=(2,12)).astype(np.float32)
    normalized=x/np.sqrt(np.mean(x*x,axis=-1,keepdims=True)+1e-6)*(1+weights['norm'])
    expected=normalized@weights['embedding'].T
    fixture=dict(checkpoint=str(checkpoint),hidden=x.ravel().tolist(),expected=expected.ravel().tolist(),
        embedding=weights['embedding'][[0,18]].ravel().tolist(),argmax=np.argmax(expected,axis=1).tolist(),input_binary=str(folder/'input.f32'))
    path=folder/'fixture.json';path.write_text(json.dumps(fixture))
    x.astype('<f4').tofile(folder/'input.f32')
    subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_cpu_io_head.py')),str(path),sys.argv[2],sys.argv[3]],check=True)
