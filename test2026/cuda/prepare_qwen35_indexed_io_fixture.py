"""Independent CPU logits, stable shortlist and BF16 projection token oracle."""
import json,sys
from pathlib import Path
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
root=Path(sys.argv[1]).resolve();output=root/'indexed-io-fixture.json';assert not output.exists()
base=json.load(open(root/'verification-io-fixture.json'));width=base['width'];checkpoint=base['checkpoint']
table=load_file(checkpoint)['model.language_model.embed_tokens.weight']
hidden=torch.tensor(base['values']).reshape(128,width);hidden[7].zero_()
logits=F.linear(hidden.to(torch.bfloat16),table).float()
cases=[]
for rows in (1,8):
    selected=[(r*17+3)%128 for r in range(rows)]
    for count in (1,16,64):
        indices=torch.argsort(logits[selected],dim=-1,descending=True,stable=True)[:,:count]
        query=hidden[:rows];subset=F.linear(query.to(torch.bfloat16),table).float()
        expected=[]
        for row in range(rows):
            ids=indices[row].tolist();maximum=max(subset[row,i].item() for i in ids)
            expected.append(min(i for i in ids if subset[row,i].item()==maximum))
        cases.append({'rows':rows,'selected':selected,'count':count,'indices':indices.flatten().tolist(),'expected':expected})
output.write_text(json.dumps({'checkpoint':checkpoint,'width':width,'vocab':table.shape[0],'values':hidden.flatten().tolist(),'cases':cases}));print(output)
