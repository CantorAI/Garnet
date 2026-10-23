"""Independent CPU BF16-head oracle for explicit multi-token verification I/O."""
import json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
root=Path(sys.argv[1]).resolve();output=root/'verification-io-fixture.json';assert not output.exists()
checkpoint=root/'mtp-component-2b-weights.safetensors'
weight=load_file(str(checkpoint))['model.language_model.embed_tokens.weight']
torch.manual_seed(84219);hidden=torch.randn(128,weight.shape[1],dtype=torch.float32)*.3
expected=F.linear(hidden.to(torch.bfloat16),weight).argmax(-1).tolist()
output.write_text(json.dumps({'checkpoint':str(checkpoint),'width':weight.shape[1],
    'values':hidden.flatten().tolist(),'expected':expected,'rows':[1,8,9,16,32,128]}));print(output)
