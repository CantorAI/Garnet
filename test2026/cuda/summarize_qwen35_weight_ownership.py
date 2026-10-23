"""Checkpoint payload by ownership category; compare with owned-runtime bytes.

This counts stored parameter bytes, not driver/context/allocator residency.
"""
import json,sys,struct
from pathlib import Path
root,output=map(Path,sys.argv[1:3]);groups={};dtypes={}
for path in sorted(root.glob('*.safetensors')):
    with path.open('rb') as f:header=json.loads(f.read(struct.unpack('<Q',f.read(8))[0]))
    for name,entry in header.items():
        if name=='__metadata__':continue
        n=entry['data_offsets'][1]-entry['data_offsets'][0]
        if name.startswith('mtp.'):group='unused_mtp'
        elif name=='model.language_model.embed_tokens.weight':group='shared_embedding_head'
        elif name.startswith('model.language_model.layers.'):group='text_layers'
        elif name.startswith('model.visual.'):group='vision'
        elif name.startswith('model.language_model.norm.'):group='final_norm'
        else:group='other'
        groups[group]=groups.get(group,0)+n;dtypes[entry['dtype']]=dtypes.get(entry['dtype'],0)+n
result={'checkpoint':str(root),'stored_parameter_bytes':groups,'dtype_bytes':dtypes,
    'scope':'Unique stored checkpoint tensors; active model excludes unused MTP, shared embedding/head counted once. Does not count runtime conversion/alignment/workspace.'}
output.write_text(json.dumps(result,indent=2));print(json.dumps(result))
