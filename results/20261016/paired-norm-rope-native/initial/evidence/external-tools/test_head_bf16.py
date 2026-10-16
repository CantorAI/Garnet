"""Compare full pretrained vocabulary logits, BF16 storage against FP32 MPS."""
import json,subprocess,sys,tempfile
from pathlib import Path
import numpy as np
runtime,build,source,model=sys.argv[1:];model=Path(model)
c=json.loads((model/'config.json').read_text());vl='text_config' in c
if vl:c=c['text_config']
prefix='model.language_model' if vl else 'model'
weight=prefix+'.embed_tokens.weight' if vl or c.get('tie_word_embeddings',False) else 'lm_head.weight'
rng=np.random.default_rng(451)
with tempfile.TemporaryDirectory(prefix='metal-bf16-head-') as temporary:
    folder=Path(temporary);cases=[]
    for batch in (1,2):
        hidden=rng.normal(size=(batch,c['hidden_size'])).astype('<f4');path=folder/('hidden-'+str(batch)+'.f32');hidden.tofile(path)
        cases.append(dict(batch=batch,hidden=str(path)))
    fixture=folder/'fixture.json';fixture.write_text(json.dumps(dict(cases=cases,width=c['hidden_size'],norm=prefix+'.norm.weight',
        weight=weight,eps=c['rms_norm_eps'],vl=vl,checkpoint=str(model/'model.safetensors'))))
    subprocess.run([runtime,str(Path(__file__).with_name('run_head_bf16.py')),str(fixture),build,source],check=True)
