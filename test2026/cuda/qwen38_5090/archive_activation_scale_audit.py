from pathlib import Path
import hashlib,json,struct,zipfile
r=Path('D:/CantorAI/5090-study-20261023');checkpoint=r/'models/Qwen3.8-27B-NVFP4';report=json.loads((r/'results/shared-activation-scale-audit.json').read_text())
ledger=r/'results/shared-activation-scale-audit-ledger';events=[json.loads(line) for line in (ledger/'commands.jsonl').read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
files=[r/'results/shared-activation-scale-audit.json',r/'audit-shared-activation-scales.py',checkpoint/'model.safetensors.index.json',checkpoint/'config.json']+[p for p in ledger.rglob('*') if p.is_file()]
if (checkpoint/'download-provenance.json').exists():files.append(checkpoint/'download-provenance.json')
raws={p.relative_to(r).as_posix():p.read_bytes() for p in files};headers={};scalar_records={}
for i,(name,row) in enumerate(report['scalars'].items()):
 shard=(checkpoint/row['shard']).resolve();assert shard.is_relative_to(checkpoint.resolve())
 with shard.open('rb') as f:
  length=struct.unpack('<Q',f.read(8))[0];assert length<=64<<20
  if row['shard'] not in headers:headers[row['shard']]=f.read(length)
  header=headers[row['shard']];assert hashlib.sha256(header).hexdigest()==row['header_sha256']
  f.seek(8+length+row['data_offsets'][0]);value=f.read(row['data_offsets'][1]-row['data_offsets'][0]);assert value.hex()==row['raw_hex']
 archive_name='scalar-bytes/'+str(i)+'.bin';raws[archive_name]=value;scalar_records[name]=archive_name
for name,raw in headers.items():raws['shard-headers/'+name+'.json']=raw
out=r/'results/shared-activation-scale-audit-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for name,raw in raws.items():z.writestr(name,raw)
 z.writestr('evidence-manifest.json',json.dumps({'scope':report['scope'],'scalar_records':scalar_records,'files':{n:hashlib.sha256(raw).hexdigest() for n,raw in raws.items()}},indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(raws)+1}))
