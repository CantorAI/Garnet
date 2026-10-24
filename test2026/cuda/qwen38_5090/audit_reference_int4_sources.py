"""Read installed package sources only; never import Torch or start GPU engines."""
from pathlib import Path
import hashlib,json,zipfile
root=Path('/mnt/d/CantorAI/5090-study-20261023');out=root/'results/reference-int4-installed-source-evidence.zip'
assert not out.exists(),'Preserve earlier source audit'
records=[];raws={};versions={}
terms=('group_size','pack-quantized','pack_quantized','symmetric','padding','pad_dim','min_thread','in_proj_ba','in_proj_a','in_proj_b','marlin','wna16')
for engine in ('vllm','sglang'):
 site=Path('/opt/cantorai5090')/engine/'lib/python3.12/site-packages';package=site/engine
 for meta in site.glob(engine+'-*.dist-info/METADATA'):
  raw=meta.read_bytes();versions[engine]={'version':next(line.split(': ',1)[1] for line in raw.decode().splitlines() if line.startswith('Version: ')),'metadata_sha256':hashlib.sha256(raw).hexdigest()}
 for path in package.rglob('*.py'):
  if 'compressed_tensors' not in path.as_posix() and 'marlin' not in path.name.lower() and 'qwen3_5' not in path.name.lower():continue
  raw=path.read_bytes();lines=raw.decode('utf8').splitlines();hits=[]
  for i,line in enumerate(lines):
   matches=[term for term in terms if term in line.lower()]
   if matches:hits.append({'line':i+1,'terms':matches,'context':'\n'.join(lines[max(0,i-2):min(len(lines),i+4)])})
  if hits:
   name='installed/'+engine+'/'+path.relative_to(package).as_posix();raws[name]=raw
   records.append({'engine':engine,'path':str(path),'archive_path':name,'sha256':hashlib.sha256(raw).hexdigest(),'hits':hits})
assert all(any(r['engine']==engine for r in records) for engine in ('vllm','sglang'))
checkpoint=root/'models/Qwen3.8-27B-INT4'
for name in ('config.json','model.safetensors.index.json'):
 raws['checkpoint/'+name]=(checkpoint/name).read_bytes()
config=json.loads(raws['checkpoint/config.json'])
report={'scope':'Installed-source discovery only. No Torch/CUDA imports, model loading, runtime compatibility, quality or performance result. Check source constraints and then perform actual startup/qualification.',
 'versions':versions,'checkpoint_quantization':config['quantization_config'],'declared_small_projection_rows':config['text_config']['linear_num_value_heads'],'files':records}
raws['report.json']=json.dumps(report,indent=2).encode();raws['audit-reference-int4.py']=Path(__file__).read_bytes()
manifest={'scope':report['scope'],'files':{name:hashlib.sha256(raw).hexdigest() for name,raw in raws.items()}}
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name,raw in raws.items():z.writestr(name,raw)
 z.writestr('evidence-manifest.json',json.dumps(manifest,indent=2))
print(json.dumps({'versions':versions,'source_files':len(records),'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest()}))
