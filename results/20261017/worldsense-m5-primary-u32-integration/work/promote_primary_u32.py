"""Select canonical WS40 only after all declared gates; restore WS10 on startup failure."""
from pathlib import Path
import datetime,hashlib,json,os,runpy,shutil,subprocess,time,urllib.error
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-primary-u32-integration')
proto=w.parent/'worldsense-primary-u32';r=Path('/Users/shawn/CantorAI');repo=r/'Garnet';n=r/'out/build/macos-arm64-release/bin'
job='gui/501/com.cantorai.cantor';plist='/Users/shawn/Library/LaunchAgents/com.cantorai.cantor.plist'
candidate=r/'out/build/garnet-metal-worldsense-primary-u32-integration-on/libgarnet.dylib'
rollback=r/'out/build/garnet-metal-worldsense-prefix-cache/libgarnet.dylib'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def command(args,check=True):return subprocess.run(args,check=check,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
def stop():
    command(['sh',str(n/'WorldSense/stop.sh')],False);command(['launchctl','bootout',job],False)
    for _ in range(150):
        if command(['launchctl','print',job],False).returncode:return
        time.sleep(.1)
    raise RuntimeError('Cantor did not unload')
def stage(path):
    temp=n/'libgarnet.dylib.selection-new';shutil.copy2(path,temp);os.replace(temp,n/'libgarnet.dylib')
def start(label):
    command(['launchctl','bootstrap','gui/501',plist])
    command([str(n/'xlang3'),'-B',str(w/'wait_cantor.py')])
    env=dict(os.environ,WORLDSENSE_STORAGE_DIR=str(r/'out/data/worldsense'));env.pop('WORLDSENSE_ENABLE_BENCHMARK',None)
    log=(w/(label+'.log')).open('ab')
    subprocess.Popen(['sh',str(n/'WorldSense/start.sh')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    a=runpy.run_path(str(w.parent/'worldsense-http.py'))
    for _ in range(60):
        try:token=a['login']();break
        except Exception:time.sleep(1)
    else:raise RuntimeError('WorldSense not ready')
    for _ in range(60):
        mounted=a['request']('/api/devices/mounted',token=token)
        if len(mounted)==4 and all(v['online']for v in mounted.values())and all('test-media/'in k for k in mounted):break
        time.sleep(1)
    else:raise RuntimeError('Four video sources did not become online')
    try:
        a['request']('/api/skill-lab/benchmark/run',method='POST',data=json.loads((w/'cases.json').read_text()),token=token)
        raise AssertionError('Benchmark remained enabled')
    except urllib.error.HTTPError as error:assert error.code==404
    return mounted

assert all(json.loads((w/'api-correctness-result.json').read_text()).values())
assert json.loads((proto/'resource-analysis.json').read_text())['passed']and json.loads((proto/'speed/analysis.json').read_text())['passed']
for folder in [proto/'confirmation',w/'production']:
    for i in [1,2]:assert json.loads((folder/('analysis-control-full-'+str(i)+'-candidate-full-'+str(i)+'.json')).read_text())['passed']
assert json.loads((w/'dependencies-after-production.json').read_text())['all_aliases_and_file_bytes_unchanged']
assert json.loads((w/'restoration-verification.json').read_text())['benchmark_valid_request_http404']
assert all(sha(repo/rel)==digest for rel,digest in json.loads((w/'canonical-source-before-build.json').read_text()).items())
env=json.loads((w/'environment.json').read_text());assert sha(candidate)==env['libraries']['on']and sha(rollback)==env['libraries']['off']and sha(n/'libgarnet.dylib')==sha(rollback)
record={'selection_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'baseline_sha256':sha(rollback),'selected_sha256':sha(candidate),'canonical_source_manifest':'canonical-source-before-build.json','canonical_commit':command(['git','-C',str(repo),'rev-parse','HEAD']).stdout.strip(),'selected_profile':'cmake/MetalWorldSense4BPrimary.cmake','scope':'Canonical prototype+actualproduction comparisons passed; normal serving selection. Live app smoke remains separate.','promoted':False}
try:
    stop();stage(candidate);mounted=start('selected-normal')
    assert sha(n/'libgarnet.dylib')==record['selected_sha256']
    record.update(promoted=True,four_file_sources_online=True,camera_unmounted=True,benchmark_valid_request_http404=True,mounted=mounted,service=command(['launchctl','print',job]).stdout)
except Exception:
    stop();stage(rollback);start('rollback-normal');record['restored_WS10_after_failure']=True
    (w/'selection-failure.json').write_text(json.dumps(record,indent=2));raise
(w/'production-selection.json').write_text(json.dumps(record,indent=2))
print('Canonical WS40 selected; normal service/fourfiles/noCam/benchmark404 verified; live smoke pending',flush=True)
