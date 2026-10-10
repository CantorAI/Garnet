"""Full-vocabulary BF16 tied-head reference without a FP32 table copy."""
import sys,json,struct,tempfile,subprocess
from pathlib import Path
import numpy as np
path=Path(sys.argv[3])/'model.safetensors'
with path.open('rb') as stream:
    length=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(length))
entry=header['model.language_model.embed_tokens.weight'];rows,dim=entry['shape']
table=np.memmap(path,dtype='<u2',mode='r',offset=8+length+entry['data_offsets'][0],shape=(rows,dim))
rng=np.random.default_rng(248320)
x=rng.normal(size=(2,dim)).astype(np.float32)
expected=np.empty((2,rows),dtype=np.float32)
for start in range(0,rows,512):
    weight=(table[start:start+512].astype(np.uint32)<<16).view(np.float32)
    expected[:,start:start+512]=x@weight.T
with tempfile.TemporaryDirectory(prefix='garnet-native-head-reference-') as temporary:
    fixture=dict(checkpoint=str(path),shape=[2,dim],input=x.ravel().tolist(),
        argmax=np.argmax(expected,axis=-1).tolist(),samples=[])
    for start in [0,128,rows//2,rows-64]:
        fixture['samples'].append(dict(start=start,expected=expected[:,start:start+64].ravel().tolist()))
    fixture_path=Path(temporary)/'fixture.json';fixture_path.write_text(json.dumps(fixture))
    subprocess.run([sys.argv[1],str(Path(__file__).with_name('run_pretrained_native_head.py')),
        str(fixture_path),sys.argv[2],sys.argv[4]],check=True)
