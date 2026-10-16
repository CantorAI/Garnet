"""Wait exact native-vector chain; then guarded component cohorts, archive/audit/decision."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();upstream_pid=int(sys.argv[3])
assert root.name=='garnet-metal-investigation'
if len(sys.argv)==4:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),str(upstream_pid),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=child.pid)));print(json.dumps(dict(pid=child.pid,output=str(out))))
else:
    assert len(sys.argv)==5 and sys.argv[4]=='--worker';begin=time.monotonic();code=1;tools=Path(__file__).parent
    def digest(path):
        with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
    def wait_exact(folder,pid,script):
        assert json.loads((folder/'pid.json').read_text())['pid']==pid
        while True:
            p=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True)
            live=p.returncode==0 and bool(p.stdout.strip())
            if live:assert script in p.stdout and str(root) in p.stdout
            (out/'observation.json').write_text(json.dumps(dict(pid=pid,script=script,live=live,terminal_status_present=(folder/'exit.json').exists(),elapsed_seconds=time.monotonic()-begin)))
            if not live:
                assert (folder/'exit.json').exists(),'Exact process missing without terminal status; do not restart'
                return json.loads((folder/'exit.json').read_text())
            time.sleep(15)
    def run(name,command):
        result=subprocess.run(command,capture_output=True,text=True)
        (out/(name+'.log')).write_text(result.stdout+result.stderr)
        (out/(name+'-command.json')).write_text(json.dumps(dict(command=command,exit_code=result.returncode),indent=2))
        assert result.returncode==0,name
        return result.stdout
    try:
        names=['queue_compact_fused_projection_groups.py','launch_compact_fused_projection_groups.py','run_compact_fused_projection_groups.py','make_compact_fused_projection_groups.py','archive_compact_fused_projection_groups.py','verify_compact_fused_projection_groups.py','analyze_compact_fused_projection_groups.py']
        identities={name:digest(tools/name) for name in names}
        for name in names:
            p=out/'controller-source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((tools/name).read_bytes())
        (out/'metadata.json').write_text(json.dumps(dict(upstream_pid=upstream_pid,source_sha256=identities,command=sys.argv,full_goal_met=False),indent=2))
        upstream=root/'logs/vector-vision-native-continuation-20261016'
        assert wait_exact(upstream,upstream_pid,'continue_vector_vision_native.py')['exit_code']==0
        assert json.loads((upstream/'result.json').read_text())['completed_cohorts'] in (['initial'],['initial','repeat'])
        labels=[]
        for label,trial in [('initial','compact-fused-projection-groups-20261016'),('repeat','compact-fused-projection-groups-confirm-20261016')]:
            folder=root/'logs'/trial
            stdout=run(label+'-launch',[sys.executable,str(tools/'launch_compact_fused_projection_groups.py'),str(root),str(folder),'component'])
            handle=json.loads(stdout.splitlines()[-1]);assert handle['output']==str(folder)
            (out/(label+'-handle.json')).write_text(json.dumps(handle,indent=2))
            status=wait_exact(folder,handle['pid'],'launch_compact_fused_projection_groups.py')
            archive=root/(trial+'.tar.gz')
            stdout=run(label+'-archive',[sys.executable,str(tools/'archive_compact_fused_projection_groups.py'),str(root),str(archive),trial])
            record=json.loads(stdout.splitlines()[-1]);assert digest(archive)==record['sha256']
            (out/(label+'-archive.json')).write_text(json.dumps(record,indent=2))
            assert status['exit_code']==0,'Failed cohort archived; no favorable rerun or partial timing admission'
            evidence=root/'validation-audits/compact-fused-projection-groups'/label
            run(label+'-extract',[sys.executable,str(root/'work/freeze_mlp_evidence.py'),str(archive),str(evidence),record['sha256']])
            run(label+'-audit',[sys.executable,str(tools/'verify_compact_fused_projection_groups.py'),str(evidence),str(evidence/'verification.json')])
            labels.append(label)
            run(label+'-decision',[sys.executable,str(tools/'analyze_compact_fused_projection_groups.py'),str(evidence.parent),*labels])
            decision=json.loads((evidence.parent/('decision-'+label+'.json')).read_text())
            print(json.dumps(dict(cohort=label,archive=record,repeat_admitted=decision['repeat_admitted'],full_goal_met=False)),flush=True)
            if label=='initial' and not decision['repeat_admitted']:break
        assert all(digest(tools/name)==sha for name,sha in identities.items())
        (out/'result.json').write_text(json.dumps(dict(completed_cohorts=labels,decision=decision,full_goal_met=False),indent=2));code=0
    finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
    sys.exit(code)
