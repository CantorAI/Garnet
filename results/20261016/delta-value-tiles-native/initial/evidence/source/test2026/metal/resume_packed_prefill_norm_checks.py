"""Reuse pinned build/CTest; retry complete guarded cases with full original coverage."""
import hashlib,json,shutil,sys
from pathlib import Path
root=Path(sys.argv[1]).resolve();prior=root/'logs/packed-prefill-norm-checks-20261014';target=Path(sys.argv[2]).resolve()
meta=json.loads((prior/'jobs/metadata.json').read_text());stages=json.loads((prior/'jobs/statuses.json').read_text())
assert json.loads((prior/'exit.json').read_text())['exit_code']==1
assert meta['candidate_sha256']=='9286535724998269f7c871e853f24ca74ff11a526e6d34c83c17c697faab44a3'
assert hashlib.file_digest((root/'build-packed-prefill-norm/libgarnet.dylib').open('rb'),'sha256').hexdigest()==meta['candidate_sha256']
for name,sha in meta['source_sha256'].items():assert hashlib.file_digest((root/'source'/name).open('rb'),'sha256').hexdigest()==sha
assert [s['name'] for s in stages[:3]]==['configure','build','ctest'] and all(s['exit_code']==0 and s['interruption'] is None for s in stages[:3])
source=(prior/'jobs.executed.py').read_text();source=source.replace('def run(name,command):','def run_once(name,command):')
source=source.replace("assert not build.exists(),'Keep existing build artifacts; use a fresh candidate directory'","assert build.is_dir(), 'Pinned candidate build disappeared'")
source=source.replace("run('configure',command);run('build',['/opt/homebrew/bin/cmake','--build',str(build),'--parallel','4'])",'')
source=source.replace("run('ctest',['/opt/homebrew/bin/ctest','--test-dir',str(build),'--output-on-failure'])",'')
source=source.replace('statuses=[];background_checks=[]','statuses=json.loads((prior/\'jobs/statuses.json\').read_text())[:3];background_checks=[]')
retry='''
def run(name,command):
    global output,statuses
    parent=output;completed=statuses
    commands=[command]
    if name.endswith('-exact-tails'):
        assert command[1].endswith('test_exact_dense_tail_reference.py')
        commands=[[command[0],str(s/'test_exact_dense_tail_case_reference.py'),*command[2:],str(i)] for i in range(8)]
    accepted=[];start=time.monotonic()
    for index,call in enumerate(commands):
        for attempt in range(20):
            quiet=None;wait_start=time.monotonic();waits=[]
            while True:
                now=time.monotonic()
                try:guard(name+'-quiet')
                except RuntimeError as error:quiet=None;waits.append(dict(elapsed_seconds=now-wait_start,reason=str(error)))
                else:
                    if quiet is None:quiet=now
                    if now-quiet>=5:break
                (parent/(name+'-quiet-waits.json')).write_text(json.dumps(waits,indent=2))
                if now-wait_start>900:raise TimeoutError('No five-second quiet validation interval')
                time.sleep(.5)
            folder=parent/'attempts'/name/('case'+str(index)+'-attempt'+str(attempt));folder.mkdir(parents=True,exist_ok=False)
            output=folder;statuses=[];failure=None
            try:run_once(name,call)
            except (AssertionError,RuntimeError) as error:failure=str(error)
            finally:
                observed=list(statuses);output=parent;statuses=completed
            if failure is not None:
                if 'Another inference runtime is active' not in failure and 'Other GPU probe PID' not in failure:raise RuntimeError('Validation failed for non-contention reason: '+failure)
                print('Discarded interrupted validation case',name,index,attempt,flush=True);continue
            assert len(observed)==1 and observed[0]['exit_code']==0
            accepted.append(dict(case_index=index,command=call,evidence=str(folder.relative_to(parent)),status=observed[0]));break
        else:raise RuntimeError('Repeated contention exhausted validation case')
    status=dict(name=name,command=command,exit_code=0,interruption=None,wall_seconds=time.monotonic()-start,accepted_cases=accepted)
    completed.append(status);(parent/'statuses.json').write_text(json.dumps(completed,indent=2))
    print('Completed all clean cases for',name,flush=True)
'''
needle="assert build.is_dir(), 'Pinned candidate build disappeared'";assert source.count(needle)==1;source=source.replace(needle,retry+'\n'+needle)
# Keep the reused stage evidence in the new job as well as in the prior archive.
needle="options=['METAL'";assert source.count(needle)==1
setup='''
for item in statuses:
    for p in (prior/'jobs').glob(item['name']+'*'):
        if p.is_file():shutil.copyfile(p,output/p.name)
(output/'statuses.json').write_text(json.dumps(statuses,indent=2))
(output/'resume-provenance.json').write_text(json.dumps(dict(prior=str(prior),candidate_sha256=meta['candidate_sha256'],reused_stages=[v['name'] for v in statuses]),indent=2))
'''
source=source.replace(needle,setup+'\n'+needle)
destination=target.with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
