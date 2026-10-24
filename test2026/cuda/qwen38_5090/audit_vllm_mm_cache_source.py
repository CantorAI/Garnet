from pathlib import Path
import hashlib,json
r=Path('/mnt/d/CantorAI/5090-study-20261023');source=Path('/opt/cantorai5090/vllm/lib/python3.12/site-packages/vllm');out=r/'results/vllm-int4-mm-cache-source-audit';out.mkdir(exist_ok=False)
paths=list((source/'entrypoints').rglob('*.py'))+list((source/'v1').rglob('*.py'))+[source/'config/multimodal.py'];report=[]
for p in paths:
 raw=p.read_bytes();text=raw.decode('utf8');lines=text.splitlines();hits=[i for i,line in enumerate(lines) if any(s in line for s in ('reset_mm_cache','reset_encoder_cache','mm_processor_cache_gb'))]
 if not hits:continue
 dest=out/'source'/p.relative_to(source);dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
 report.append({'path':p.relative_to(source).as_posix(),'sha256':hashlib.sha256(raw).hexdigest(),'hits':[{'line':i+1,'context':lines[max(0,i-3):i+7]} for i in hits]})
(out/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps([{'path':x['path'],'lines':[h['line'] for h in x['hits']]} for x in report]))
