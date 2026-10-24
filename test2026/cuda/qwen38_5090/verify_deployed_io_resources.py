from pathlib import Path
import hashlib,io,json,zipfile
out=Path(__file__).parent;archive=out/'deployed-io-resources-evidence.zip'
with zipfile.ZipFile(archive) as z:
 assert z.testzip() is None;e=json.loads(z.read('evidence-manifest.json'))
 for n,h in e['files'].items():assert hashlib.sha256(z.read(n)).hexdigest()==h,n
 for n,h in e['linked_archives'].items():assert hashlib.sha256((out/n).read_bytes()).hexdigest()==h,n
 assert e['library_sha256']=='ba67dc9d4f8b9948615e522e3b6ad0aa10ea6f742505465084449fe57ead0b8b'
 for name,exit in [('deployed-io-resources-controller-ledger',1),('deployed-io-resources-parse-ledger',0)]:
  ev=[json.loads(l) for l in z.read('results/'+name+'/commands.jsonl').decode().splitlines()];assert ev[-1]['event']=='finished' and ev[-1]['exit_code']==exit
 base='results/deployed-io-resources-attempt1/';events=[json.loads(l) for l in z.read(base+'resource-ledger/commands.jsonl').decode().splitlines()];assert events[-1]['exit_code']==0
 log=z.read(base+'resource-ledger/commands.log');begin,end=events[-1]['raw_start_offset'],events[-1]['raw_end_offset'];raw=log[begin:end].decode('utf8');arch=None;pending=None;rows=[]
 for line in raw.splitlines():
  line=line.strip()
  if line.startswith('arch = '):arch=line.split('=',1)[1].strip()
  elif line.startswith('Function '):
   name=line.removeprefix('Function ').strip().rstrip(':');pending=name if any(k in name for k in ('GatherChunk','WriteRequest')) else None
  elif pending and line.startswith('REG:'):
   values=dict(token.split(':',1) for token in line.split() if ':' in token)
   rows.append({'function':pending,'architecture':arch,'local_bytes_per_thread':int(values['LOCAL']),'stack_bytes_per_thread':int(values['STACK']),'constant0_bytes':int(values['CONSTANT[0]'])});pending=None
 assert len(rows)==2 and all(v['architecture']=='sm_120' for v in rows)
 report=json.loads(z.read(base+'corrected-parse.json'));assert report['library_sha256']==e['library_sha256'] and report['resource_log_sha256']==hashlib.sha256(log).hexdigest()
 assert [{k:v for k,v in row.items() if k!='raw'} for row in report['kernels']]==rows
 with zipfile.ZipFile(io.BytesIO(z.read('cuobjdump132-tool.zip'))) as tool:
  receipt=json.loads(tool.read('manifest.json'));actual=hashlib.sha256(tool.read('cuobjdump.exe')).hexdigest();assert actual==receipt['sha256']=='7405a7a6f0c64bc7d962bb5cfc30ce24d122c8c6ac6d89c0636f12f60bf3c6fe'
 signature=json.loads(z.read(base+'signature.json'));assert signature['Status']=='Valid' and 'NVIDIA Corporation' in signature['Signer']
 for n,h in events[0]['artifact_sha256'].items():
  name=n.replace('\\','/').split('5090-study-20261023/',1)[1]
  if name=='build/bin/garnet.dll':assert h==e['library_sha256']
  elif name=='tools/cuobjdump132/cuobjdump.exe':assert h==actual
  else:assert hashlib.sha256(z.read(name)).hexdigest()==h,name
 result={'status':'deployed_io_resource_evidence_verified','archive_sha256':hashlib.sha256(archive.read_bytes()).hexdigest(),'files_verified':len(e['files']),'library_sha256':e['library_sha256'],'kernels':rows,'parameter_local_or_stack_copy_hypothesis_supported':any(v['local_bytes_per_thread'] or v['stack_bytes_per_thread'] for v in rows),'scope':'Independent CRC/SHA, known signed-tool byte digest, terminal/failure evidence and line-state replay of exact raw CUDA resource output. Remote Authenticode is a retained producer receipt. Deployed DLL bytes remain remote; artifact hashes link the tool input. Zero static LOCAL/STACK does not attribute total VRAM or explain pauses. No GPU timing, source/binary/driver change, speed or2x claim.'}
(out/'deployed-io-resources-verified.json').write_text(json.dumps(result,indent=2),encoding='utf8');print(json.dumps(result,indent=2))
