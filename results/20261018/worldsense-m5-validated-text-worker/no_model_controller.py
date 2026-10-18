"""Sequential integrated native gateway/validated child no-model checks."""
from pathlib import Path
import subprocess,sys,json,hashlib,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');n=r/'out/build/macos-arm64-release/bin';s=r/'out/research/worldsense-validated-text-worker/source'
m=json.loads((w/'gateway-source-manifest.json').read_text());assert all(hashlib.sha256((s/k).read_bytes()).hexdigest()==v for k,v in m.items())
identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest()for p in [w/'gateway_nomodel_guard.py',Path(__file__).resolve(),n/'xlang3',n/'libxlang3_runtime.dylib']}
(w/'no-model-identities-before.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),files=identity,libraries={mode:{name:hashlib.sha256((r/('out/build/garnet-metal-worldsense-validated-text-worker-'+mode)/name).read_bytes()).hexdigest()for name in ['libgarnet.dylib','libgarnet_worker_host.dylib']}for mode in ['on','primary','fault']}),indent=2)+'\n')
for mode in ['on','primary','fault']:
 with(w/('gateway-nomodel-'+mode+'.log')).open('w')as f:
  result=subprocess.run([str(n/'xlang3'),'-B',str(w/'gateway_nomodel_guard.py'),mode],stdout=f,stderr=subprocess.STDOUT,timeout=120)
 print('Native validated-worker no-model guard '+mode+' exit '+str(result.returncode),flush=True)
 assert result.returncode==0,'No-model guard failed; retained log and before-assert responses'
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in identity.items())
(w/'gateway-nomodel-result.json').write_text(json.dumps(dict(all_three_integrated_parent_roles_pass=True,exact_old_backend_and_new_host_provenance_pass=True,owned_error_no_hidden_retry_explicit_reload_shutdown_pass=True,no_model_inference_or_speed_result=True),indent=2)+'\n')
print('WS61 three-role integrated native gateway/validated-child plumbing PASS; no model/speed claim',flush=True)
