"""Read-only final normal stack verification; private authentication never exported."""
from pathlib import Path
import subprocess,os,runpy,json,hashlib,re,urllib.error,datetime,sys
root=Path('/Users/shawn/CantorAI');n=root/'out/build/macos-arm64-release/bin';base=Path('/Users/shawn/Documents/Codex/2026-10-15/i');slug=sys.argv[1];commit=sys.argv[2];w=base/'work'/slug
expected='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72';assert hashlib.sha256((n/'libgarnet.dylib').read_bytes()).hexdigest()==expected
subprocess.run([str(n/'xlang3'),'-B',str(w/'wait_cantor.py')],check=True)
a=runpy.run_path(str(base/'work/worldsense-http.py'));token=a['login']();mounted=a['request']('/api/devices/mounted',token=token);assert len(mounted)==4 and all(v['online']for v in mounted.values())and all('test-media/'in k for k in mounted)
try:a['request']('/api/skill-lab/benchmark/run',method='POST',data=json.loads((w/'cases.json').read_text()),token=token);raise AssertionError('Benchmark remained enabled')
except urllib.error.HTTPError as e:assert e.code==404
text=subprocess.check_output(['launchctl','print','gui/501/com.cantorai.cantor'],text=True);pid=int(re.search(r'\n\s+pid = (\d+)',text).group(1));env=dict(os.environ,PATH='/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin')
assert subprocess.check_output(['git','-C',str(root/'Garnet'),'rev-parse','HEAD'],text=True,env=env).strip()==commit
assert not subprocess.check_output(['git','-C',str(root/'Garnet'),'status','--short'],text=True,env=env)
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=pid,WS10_sha256=expected,four_file_sources_online=True,no_camera=True,valid_benchmark_post_http404=True,commit=commit,repository_clean=True);(base/'work/worldsense-opt/ws59-normal-final.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))
