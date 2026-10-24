from pathlib import Path
import datetime,json,os,signal
r=Path('/mnt/d/CantorAI/5090-study-20261023');out=r/'results/vllm-int4-cache-reset-self-qualified-intentional-stop.json';assert not out.exists()
pilot=r/'results/vllm-int4-cache-reset-self-qualified-pilot';completion=json.loads((pilot/'completed.json').read_text());assert completion['status']=='completed' and completion['cache_counter_delta']['hit_delta']==0
files=list((pilot/'benchmark').glob('*-manifest.json'));assert len(files)==1 and json.loads(files[0].read_text())['status']=='completed'
for label in ('quality','self-qualified-http'):
 events=[json.loads(line) for line in (r/('results/vllm-int4-cache-reset-'+label+'-ledger/commands.jsonl')).read_text().splitlines()]
 assert events[-1]['event']=='finished' and events[-1]['exit_code']==0
matches=[]
for p in Path('/proc').glob('[0-9]*/cmdline'):
 try:raw=p.read_bytes();args=raw.decode().strip('\0').split('\0')
 except (OSError,UnicodeError):continue
 if not args or not args[0].startswith('/opt/cantorai5090/vllm/bin/'):continue
 if '/mnt/d/CantorAI/5090-study-20261023/models/Qwen3.8-27B-INT4' not in args or 'serve' not in args:continue
 if '--served-model-name' not in args or args[args.index('--served-model-name')+1]!='qwen38-27b-int4':continue
 if '--port' not in args or args[args.index('--port')+1]!='18080':continue
 matches.append((p,raw,args))
assert len(matches)==1,'Ambiguous reference server identity'
p,raw,args=matches[0];selected={entry.split(b'=',1)[0].decode():entry.split(b'=',1)[1].decode() for entry in (p.parent/'environ').read_bytes().split(b'\0') if entry.startswith((b'VLLM_SERVER_DEV_MODE=',b'GARNET_VLLM_LOG='))};assert selected.get('VLLM_SERVER_DEV_MODE')=='1';pid=int(p.parent.name);record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':pid,'args':args,'selected_environment':selected,'stat':(p.parent/'stat').read_text(),'reason':'Intentional study-owned SIGTERM after current-process-qualified exploratory cache-reset pilot; separate-process gate remains failed and preserved; exit still needs observation','signal':'SIGTERM'}
os.kill(pid,signal.SIGTERM);out.write_text(json.dumps(record,indent=2));print('IDENTITY_GUARDED_VLLM_INT4_SIGTERM_SENT',pid)
