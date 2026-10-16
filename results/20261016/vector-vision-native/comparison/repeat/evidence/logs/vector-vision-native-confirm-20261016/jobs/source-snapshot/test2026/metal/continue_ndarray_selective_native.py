"""Wait exact full30 controller, then guarded native cohorts, archive/audit, and predeclared decision."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();upstream_pid=int(sys.argv[3])
assert root.name=='ndarray-selective-native-packed-investigation-20261016'
if len(sys.argv)==4:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),str(upstream_pid),'--worker'],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
    assert len(sys.argv)==5 and sys.argv[4]=='--worker';begin=time.monotonic();code=1
    parent=Path(json.loads((root/'preparation.json').read_text())['parent']);tools=Path(__file__).parent
    def digest(p):
        with p.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def wait_exact(folder,pid,script):
        assert json.loads((folder/'pid.json').read_text())['pid']==pid
        while True:
            p=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
            live=p.returncode==0 and bool(p.stdout.strip())
            if live:assert script in p.stdout and str(root) in p.stdout
            (out/'observation.json').write_text(json.dumps(dict(pid=pid,script=script,live=live,
                terminal_status_present=(folder/'exit.json').exists(),elapsed_seconds=time.monotonic()-begin)))
            if not live:
                assert (folder/'exit.json').exists(),'Exact process missing without status; do not restart'
                return json.loads((folder/'exit.json').read_text())
            time.sleep(15)
    def run(name,command):
        p=subprocess.run(command,capture_output=True,text=True)
        (out/(name+'.log')).write_text(p.stdout+p.stderr)
        (out/(name+'-command.json')).write_text(json.dumps(dict(command=command,exit_code=p.returncode),indent=2))
        assert p.returncode==0,name
        return p.stdout
    try:
        names=['continue_ndarray_selective_native.py','launch_ndarray_selective_native_profiles.py',
            'run_ndarray_selective_native_profiles.py','compare_ndarray_selective_native.py',
            'archive_ndarray_selective_native_profiles.py','verify_ndarray_selective_native_profiles.py',
            'analyze_ndarray_selective_native_results.py']
        identities={name:digest(tools/name) for name in names}
        for name in names:
            dest=out/'controller-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((tools/name).read_bytes())
        (out/'metadata.json').write_text(json.dumps(dict(upstream_pid=upstream_pid,source_sha256=identities,command=sys.argv,
            scope=__doc__,full_goal_met=False),indent=2))
        upstream=parent/'logs/ndarray-selective-native-finalize-20261016'
        assert wait_exact(upstream,upstream_pid,'finalize_ndarray_selective_native_checks.py')['exit_code']==0
        assert json.loads((upstream/'result.json').read_text())['independent_audit_passed']
        labels=[]
        for label,trial in [('initial','ndarray-selective-native-20261016'),('repeat','ndarray-selective-native-confirm-20261016')]:
            folder=root/'logs'/trial
            stdout=run(label+'-launch',[sys.executable,str(tools/'launch_ndarray_selective_native_profiles.py'),str(root),str(folder),'profiles'])
            handle=json.loads(stdout.splitlines()[-1]);assert handle['output']==str(folder)
            (out/(label+'-handle.json')).write_text(json.dumps(handle,indent=2))
            status=wait_exact(folder,handle['pid'],'launch_ndarray_selective_native_profiles.py')
            archive=parent/(trial+'.tar.gz')
            stdout=run(label+'-archive',[sys.executable,str(tools/'archive_ndarray_selective_native_profiles.py'),str(root),str(archive),trial])
            frozen=json.loads(stdout.splitlines()[-1]);assert digest(archive)==frozen['sha256']
            (out/(label+'-archive.json')).write_text(json.dumps(frozen,indent=2))
            assert status['exit_code']==0,'Failed native cohort archived; do not restart or admit partial timings'
            evidence=root/'validation-audits/ndarray-selective-native-results'/label
            run(label+'-extract',[sys.executable,str(parent/'work/freeze_mlp_evidence.py'),str(archive),str(evidence),frozen['sha256']])
            run(label+'-audit',[sys.executable,str(tools/'verify_ndarray_selective_native_profiles.py'),str(evidence),str(evidence/'verification.json'),trial])
            labels.append(label)
            run(label+'-decision',[sys.executable,str(tools/'analyze_ndarray_selective_native_results.py'),str(evidence.parent),*labels])
            decision=json.loads((evidence.parent/('decision-'+label+'.json')).read_text())
            print(json.dumps(dict(cohort=label,archive=frozen,repeat_admitted=decision['repeat_admitted'],full_goal_met=False)),flush=True)
            if label=='initial' and not decision['repeat_admitted']:break
        assert all(digest(tools/name)==sha for name,sha in identities.items())
        (out/'result.json').write_text(json.dumps(dict(completed_cohorts=labels,decision=decision,full_goal_met=False),indent=2));code=0
    finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
    sys.exit(code)
