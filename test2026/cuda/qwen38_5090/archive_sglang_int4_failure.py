from pathlib import Path
import hashlib,json,zipfile
r=Path('D:/CantorAI/5090-study-20261023');ledger=r/'results/sglang-int4-server-ledger/commands.jsonl';events=[json.loads(s) for s in ledger.read_text().splitlines()];assert events[-1]['event']=='finished' and events[-1]['exit_code']!=0
log=r/'results/sglang-int4-server.log';raw=log.read_bytes();assert b'size_n = 96 is not divisible by tile_n_size = 64' in raw
files=list((r/'results/sglang-int4-server-ledger').glob('*'))+[log,r/'Garnet/test2026/cuda/qwen38_5090/serve_sglang_int4.sh',r/'results/int4-reference-source-install.json',r/'study-state.json',r/'preparation-manifest.json',r/'models/Qwen3.8-27B-INT4/config.json',r/'models/Qwen3.8-27B-INT4/model.safetensors.index.json',r/'results/reference-int4-installed-source-evidence.zip']
report={'status':'stock_installed_load_failed','exit_code':events[-1]['exit_code'],'specific_failure':'Marlin merged GDN projection N96 not divisible by tileN64','scope':'Installed SGLang0.5.22 pinned compressed-tensors INT4 launch only; no inference, quality or benchmark. Not a universal SGLang unsupported claim. No checkpoint/precision substitution.'}
(r/'results/sglang-int4-load-failure.json').write_text(json.dumps(report,indent=2));files.append(r/'results/sglang-int4-load-failure.json');files.append(r/'archive-sglang-int4-failure.py')
blobs={p.relative_to(r).as_posix():p.read_bytes() for p in files if p.is_file()};out=r/'results/sglang-int4-load-failure-evidence.zip';assert not out.exists()
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
 for n,b in blobs.items():z.writestr(n,b)
 z.writestr('evidence-manifest.json',json.dumps({'files':{n:hashlib.sha256(b).hexdigest() for n,b in blobs.items()},'report':report},indent=2))
print(json.dumps({'bytes':out.stat().st_size,'sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'entries':len(blobs)+1,'report':report}))
