"""Independent embedding/tied-head expectations for native CUDA request I/O."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from safetensors.torch import save_file
out=Path(sys.argv[1]).resolve();out.mkdir(parents=True,exist_ok=True)
rng=np.random.default_rng(935)
table=torch.from_numpy(rng.normal(0,.2,(32,8)).astype(np.float32)).to(torch.bfloat16)
save_file({'model.language_model.embed_tokens.weight':table},str(out/'model.safetensors'))
x=torch.from_numpy(rng.normal(0,.3,(2,8)).astype(np.float32))
logits=torch.nn.functional.linear(x.to(torch.bfloat16),table).float()
fixture={'checkpoint':str(out),'tokens':[3,17],'embedding':table[[3,17]].float().ravel().tolist(),
 'hidden':x.ravel().tolist(),'logits':logits.ravel().tolist(),'predictions':logits.argmax(-1).tolist()}
(out/'fixture.json').write_text(json.dumps(fixture));print(out/'fixture.json')
