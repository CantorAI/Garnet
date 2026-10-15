"""Reuse the exact archived build/CTest; preserve rejected outputs before retries."""
from pathlib import Path
import hashlib,json,shutil,sys

root=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve()
prior=root/'logs/mlp-prefill-native-checks-guardfix-20261015'
assert json.loads((prior/'exit.json').read_text())['exit_code']==1
with (root/'mlp-prefill-native-checks-guardfix-20261015.tar.gz').open('rb') as stream:
    assert hashlib.file_digest(stream,'sha256').hexdigest()=='6d9fda15edcd9bf2e101918ed7615925b8f3d5f98cfc016ec34cd1da806b2147'
meta=json.loads((prior/'jobs/metadata.json').read_text())
manifest=json.loads((root/'mlp-prefill-native-staging/manifest.json').read_text());assert len(manifest)==18
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in manifest.items())
assert all(hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha for name,sha in meta['source_sha256'].items())
build=root/'build-mlp-prefill-graph-guardfix'
assert hashlib.file_digest((build/'libgarnet.dylib').open('rb'),'sha256').hexdigest()==meta['candidate_sha256']
stages=json.loads((prior/'jobs/statuses.json').read_text());assert len(stages)==3
assert [v['name'] for v in stages]==['configure','build','ctest']
assert all(v['exit_code']==0 and v['interruption'] is None and len(v['accepted_cases'])==1 for v in stages)
source=(prior/'jobs.executed.py').read_text()
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old));source=source.replace(old,new)
replace("assert not build.exists(),'Keep existing build artifacts; use a fresh candidate directory'", "assert build.is_dir(), 'Pinned MLP build disappeared'")
replace("run('configure',command);run('build',['/opt/homebrew/bin/cmake','--build',str(build),'--parallel','4'])",'')
replace("run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])",'')
replace('statuses=[];background_checks=[]',"statuses=json.loads((prior/'jobs/statuses.json').read_text());background_checks=[]")
replace("                print('Discarded interrupted validation case',name,index,attempt,flush=True);continue",'''                # run_once has terminated and waited for its owned group. Keep
                # every partial output under this rejected attempt, then retry
                # the unchanged command with a fresh output path.
                relocated=[]
                for argument in call[2:]:
                    path=Path(argument)
                    if not path.is_absolute():continue
                    path=path.resolve()
                    if path==parent or not path.is_relative_to(parent) or not path.exists():continue
                    destination=folder/'rejected-artifacts'/path.relative_to(parent)
                    assert destination.resolve().is_relative_to(folder.resolve()) and not destination.exists()
                    destination.parent.mkdir(parents=True,exist_ok=True);path.rename(destination)
                    relocated.append(dict(original=str(path),preserved=str(destination)))
                (folder/'rejected-artifacts.json').write_text(json.dumps(relocated,indent=2)+'\\n')
                print('Discarded interrupted validation case',name,index,attempt,flush=True);continue''')
needle="options=['METAL'";assert source.count(needle)==1
setup='''
for item in statuses:
    for case in item['accepted_cases']:
        source_case=(prior/'jobs'/case['evidence']).resolve();destination=(output/case['evidence']).resolve()
        assert source_case.is_relative_to(prior/'jobs') and destination.is_relative_to(output)
        destination.parent.mkdir(parents=True,exist_ok=True);shutil.copytree(source_case,destination)
(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
(output/'resume-provenance.json').write_text(json.dumps(dict(prior=str(prior),candidate_sha256=meta['candidate_sha256'],reused_stages=[v['name'] for v in statuses],prior_archive_sha256='6d9fda15edcd9bf2e101918ed7615925b8f3d5f98cfc016ec34cd1da806b2147',source_manifest=manifest),indent=2))
'''
source=source.replace(needle,setup+'\n'+needle)
destination=target.with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
