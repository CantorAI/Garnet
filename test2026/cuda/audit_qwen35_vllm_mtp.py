"""Read-only installed-vLLM MTP capability evidence; never imports torch/vLLM."""
import hashlib,importlib.metadata,json,re,sys
from pathlib import Path
root=Path(sys.prefix)/'lib'/('python'+str(sys.version_info.major)+'.'+str(sys.version_info.minor))/'site-packages/vllm'
pattern=re.compile(r'Qwen3_5MTP|qwen3_5_mtp|Qwen3_5.*MTP|Qwen3_5.*NextN')
files=[]
for path in sorted(root.rglob('*.py')):
    source=path.read_text(errors='replace');lines=source.splitlines();hits=[]
    for index,line in enumerate(lines):
        if pattern.search(line):hits.append({'line':index+1,'context':lines[max(0,index-3):index+5]})
    if hits:files.append({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'hits':hits})
spec=root/'config/speculative.py';selected=[]
if spec.exists():
    lines=spec.read_text().splitlines()
    for i,line in enumerate(lines):
        if any(word in line for word in ('qwen3_5','qwen3.5','Qwen3_5','"mtp"','num_speculative_tokens')):
            selected.append({'line':i+1,'context':lines[max(0,i-2):i+4]})
result={'versions':{name:importlib.metadata.version(name) for name in ('vllm','torch','transformers')},
    'root':str(root),'matching_files':files,'speculative_config_source':str(spec),
    'speculative_config_sha256':hashlib.sha256(spec.read_bytes()).hexdigest() if spec.exists() else None,
    'selected_speculative_config':selected,
    'scope':'Installed source capability audit only; no model/GPU import, no successful MTP runtime or performance acceptance.'}
with open(sys.argv[1],'x') as stream:json.dump(result,stream,indent=2)
print(json.dumps({'versions':result['versions'],'matching_files':[f['path'] for f in files],'output':sys.argv[1]},indent=2))
