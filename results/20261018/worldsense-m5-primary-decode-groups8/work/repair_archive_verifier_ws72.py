"""Retain first canonical audit failure; remove unused local build binding only."""
from pathlib import Path
import json,shutil,hashlib,datetime,ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt';w=b/'work/worldsense-primary-decode-groups8';d=Path('/Users/shawn/CantorAI/Garnet/results/20261018/worldsense-m5-primary-decode-groups8');a=w/'canonical-verifier-local-binding-review';a.mkdir(exist_ok=False)
for p in [f/'verify_archive_ws72.py',f/'prepare_archive_verifier_ws72.py',w/'canonical-independent-audit.log']:shutil.copy2(p,a/p.name)
before=" build=r/('out/build/garnet-metal-worldsense-primary-decode-groups8-'+role);";after=' '
p=f/'verify_archive_ws72.py';t=p.read_text();assert t.count(before)==1;t=t.replace(before,after);ast.parse(t);p.write_text(t)
p=f/'prepare_archive_verifier_ws72.py';t=p.read_text();needle="needle='programs={role:{kind:';";assert t.count(needle)==1;t=t.replace(needle,'new=new.replace('+repr(before)+','+repr(after)+')\n'+needle);ast.parse(t);p.write_text(t)
note='First canonical audit exited1: an unused local build=r/... assignment copied from the standalone numerical verifier referenced r, which is intentionally absent in archive-only verification. All actual native word reads already use d/native-* archived files. Remove ONLY that unused assignment; no data path/assertion/source/native/model/metric/threshold change. Original failed verifier/generator/log retained. Recompute manifest after adding this erratum and fixed verification tools; reuse same32pipeline calls and all native/learned evidence, no inference/timing rerun.\n'
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),observed_failed_exit_code=1,error='NameError r in unused local build binding before group word checks',before=before,after=after,scope=note,source_model_native_data_metric_gate_bytes_unchanged=True,no_new_inference_or_timing=True)
(a/'erratum.json').write_text(json.dumps(record,indent=2)+'\n');shutil.copytree(a,d/'canonical-verifier-local-binding-review')
for name in ['verify_archive_ws72.py','prepare_archive_verifier_ws72.py','repair_archive_verifier_ws72.py']:shutil.copy2(f/name,d/'work'/name)
for root in [w,d]:
 p=root/('README.md'if root==d else'result.txt');p.write_text(p.read_text()+'\nCanonical archive verifier erratum\n'+note)
for name in ['Garnet-4B-WS72-Result.txt','Garnet-4B-Latest-Result.txt','4B-Latest-Result.txt']:
 p=b/'outputs'/name;p.write_text(p.read_text()+'\nCanonical archive verifier erratum\n'+note)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();(d/'file-manifest.json').write_text(json.dumps({str(p.relative_to(d)):sha(p)for p in sorted(d.rglob('*'))if p.is_file()and p.name!='file-manifest.json'},indent=2)+'\n');print('Canonical unused binding repair/firstfailure retained; no measured source/data/gates changed. Manifest entries',len(json.loads((d/'file-manifest.json').read_text())))
