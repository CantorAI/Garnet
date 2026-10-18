from pathlib import Path
import subprocess,os,time,json,runpy,urllib.error,hashlib,shutil,re
w=Path(__file__).resolve().parent;n=Path('/Users/shawn/CantorAI/out/build/macos-arm64-release/bin');job='gui/501/com.cantorai.cantor';plist='/Users/shawn/Library/LaunchAgents/com.cantorai.cantor.plist'
subprocess.run(['sh',str(n/'WorldSense/stop.sh')],check=False);subprocess.run(['launchctl','bootout',job],check=False)
for _ in range(150):
 if subprocess.run(['launchctl','print',job],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode:break
 time.sleep(.1)
else:raise RuntimeError('Cantor did not unload')
build=Path('/Users/shawn/CantorAI/out/build/garnet-main-macos-verify-20261018');library=build/'libgarnet.dylib';temp=n/'libgarnet.dylib.main-deployment';shutil.copy2(library,temp);os.replace(temp,n/'libgarnet.dylib')
assert hashlib.sha256((n/'libgarnet.dylib').read_bytes()).digest()==hashlib.sha256(library.read_bytes()).digest()
assert subprocess.check_output(['git','branch','--show-current'],cwd='/Users/shawn/CantorAI/Garnet',text=True).strip()=='main'
subprocess.run(['launchctl','bootstrap','gui/501',plist],check=True)
subprocess.run([str(n/'xlang3'),'-B',str(w/'wait_cantor.py')],check=True)
env=dict(os.environ,WORLDSENSE_STORAGE_DIR='/Users/shawn/CantorAI/out/data/worldsense');env.pop('WORLDSENSE_ENABLE_BENCHMARK',None)
f=(w/'main-deployed-normal.log').open('ab');subprocess.Popen(['sh',str(n/'WorldSense/start.sh')],env=env,stdout=f,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
a=runpy.run_path(str(w.parent/'worldsense-http.py'))
for _ in range(60):
 try:t=a['login']();break
 except Exception:time.sleep(1)
else:raise RuntimeError('normal server not ready')
d=a['request']('/api/devices/mounted',token=t);assert len(d)==4 and all(v['online']for v in d.values())and all('test-media/'in k for k in d)
try:
 a['request']('/api/skill-lab/benchmark/run',method='POST',data=json.loads((w/'cases.json').read_text()),token=t)
 raise AssertionError('Benchmark remained enabled')
except urllib.error.HTTPError as e:assert e.code==404
(w/'main-deployment-verification.json').write_text(json.dumps(dict(validated_main_binary_deployed=True,library_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),main_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd='/Users/shawn/CantorAI/Garnet',text=True).strip(),launchd_pid=int(re.search(r'\bpid = (\d+)',subprocess.check_output(['launchctl','print',job],text=True)).group(1)),four_file_sources_online=True,camera_unmounted=True,benchmark_valid_request_http404=True,mounted=d),indent=2));print('Verified main deployed, four files online, camera unmounted, benchmark404 PASS',flush=True)
