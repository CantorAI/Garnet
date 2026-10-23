"""Combine existing fixed synthetic target/MTP tensors, without new directories."""
import json,sys
from pathlib import Path
from safetensors.torch import load_file,save_file
root=Path(sys.argv[1]).resolve();output=root/'speculative-session-fixture.json';assert not output.exists()
base=json.loads((root/'decoder-fixture-b4-real-attention/fixture.json').read_text())
weights=load_file(str(Path(base['checkpoint'])/'model.safetensors'))
mtp=load_file(str(root/'mtp-component-4b-weights.safetensors'))
weights.update({k:v for k,v in mtp.items() if k.startswith('mtp.')})
checkpoint=root/'speculative-session-fixture.safetensors';assert not checkpoint.exists();save_file(weights,str(checkpoint))
output.write_text(json.dumps({'checkpoint':str(checkpoint),'config':base['config'],'capacity':32,
    'prompt_steps':[s['tokens'] for s in base['steps']],'scope':'Fixed synthetic target+MTP correctness; not performance'}));print(output)
