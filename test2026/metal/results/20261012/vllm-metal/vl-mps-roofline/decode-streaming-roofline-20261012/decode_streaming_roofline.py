import json,struct
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation');reports=[]
for name in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
 model=root/'models'/name;c=json.loads((model/'config.json').read_text());vl='text_config' in c
 if vl:c=c['text_config']
 head='model.language_model.embed_tokens.weight' if vl else 'model.embed_tokens.weight' if c.get('tie_word_embeddings') else 'lm_head.weight'
 prefix='model.language_model.layers.' if vl else 'model.layers.'
 with (model/'model.safetensors').open('rb') as stream:n=struct.unpack('<Q',stream.read(8))[0];header=json.loads(stream.read(n))
 tensors={k:v['data_offsets'][1]-v['data_offsets'][0] for k,v in header.items() if k!='__metadata__' and (k.startswith(prefix) or k==head)}
 bf16_matrix_bytes=sum(tensors[k] for k in tensors if len(header[k]['shape'])==2 and header[k]['dtype']=='BF16')
 reports.append(dict(model=name,decoder_and_head_checkpoint_bytes=sum(tensors.values()),bf16_projection_head_bytes=bf16_matrix_bytes,
  streaming_120GBs_steps_per_second=120e9/bf16_matrix_bytes,tensors=tensors,
  scope='Streaming roofline only, not a universal algorithmic lower bound: assumes each dense projection/head BF16 value is read once per decode step; excludes cache traffic and arithmetic. Lossless packing/pruning can change the premise.'))
output=root/'logs/decode-streaming-roofline-20261012';output.mkdir(parents=True,exist_ok=True)
(output/'report.json').write_text(json.dumps(dict(hardware_bandwidth_source='https://support.apple.com/en-gb/121555',advertised_bytes_per_second=120000000000,models=reports),indent=2)+'\n')
for r in reports:print(r['model'],r['bf16_projection_head_bytes'],r['streaming_120GBs_steps_per_second'])
