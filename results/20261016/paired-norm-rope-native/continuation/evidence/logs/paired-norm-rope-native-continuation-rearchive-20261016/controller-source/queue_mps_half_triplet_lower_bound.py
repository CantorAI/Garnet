"""Wait exact compact component chain; preserve one terminal synthetic prefill feasibility probe."""
from pathlib import Path
import hashlib,json,subprocess,sys,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve();upstream_pid=int(sys.argv[3])
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
            p=subprocess.run(['ps','-p',str(pid),'-o','command='],capture_output=True,text=True);live=p.returncode==0 and bool(p.stdout.strip())
            if live:assert script in p.stdout and str(root) in p.stdout
            (out/'observation.json').write_text(json.dumps(dict(pid=pid,script=script,live=live,terminal_status_present=(folder/'exit.json').exists(),elapsed_seconds=time.monotonic()-begin)))
            if not live:
                assert (folder/'exit.json').exists(),'Missing exact process without status; do not restart'
                return json.loads((folder/'exit.json').read_text())
            time.sleep(15)
    def run(label,command):
        p=subprocess.run(command,capture_output=True,text=True);(out/(label+'.log')).write_text(p.stdout+p.stderr)
        (out/(label+'-command.json')).write_text(json.dumps(dict(command=command,exit_code=p.returncode),indent=2));assert p.returncode==0,label;return p.stdout
    try:
        names=['queue_mps_half_triplet_lower_bound.py','launch_mps_half_triplet_lower_bound.py','run_mps_half_triplet_lower_bound.py','mps_half_triplet_lower_bound.mm','archive_mps_half_triplet_lower_bound.py','verify_mps_half_triplet_lower_bound.py']
        identities={name:digest(tools/name) for name in names}
        for name in names:
            dest=out/'controller-source'/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((tools/name).read_bytes())
        (out/'metadata.json').write_text(json.dumps(dict(upstream_pid=upstream_pid,source_sha256=identities,command=sys.argv,full_goal_met=False),indent=2))
        upstream=root/'logs/compact-fused-projection-groups-queue-20261016'
        assert wait_exact(upstream,upstream_pid,'queue_compact_fused_projection_groups.py')['exit_code']==0
        trial='mps-half-triplet-lower-bound-20261016';folder=root/'logs'/trial
        stdout=run('launch',[sys.executable,str(tools/'launch_mps_half_triplet_lower_bound.py'),str(root),str(folder),'capability']);handle=json.loads(stdout.splitlines()[-1]);assert handle['output']==str(folder)
        (out/'handle.json').write_text(json.dumps(handle,indent=2));status=wait_exact(folder,handle['pid'],'launch_mps_half_triplet_lower_bound.py')
        archive=root/(trial+'.tar.gz');stdout=run('archive',[sys.executable,str(tools/'archive_mps_half_triplet_lower_bound.py'),str(root),str(archive),trial]);record=json.loads(stdout.splitlines()[-1]);assert record['sha256']==digest(archive)
        (out/'archive.json').write_text(json.dumps(record,indent=2))
        if status['exit_code']==0:
            evidence=root/'validation-audits'/trial
            run('extract',[sys.executable,str(root/'work/freeze_mlp_evidence.py'),str(archive),str(evidence),record['sha256']])
            run('audit',[sys.executable,str(tools/'verify_mps_half_triplet_lower_bound.py'),str(evidence),str(evidence/'verification.json')]);code=0
        else:code=2
        assert all(digest(tools/name)==sha for name,sha in identities.items())
        (out/'result.json').write_text(json.dumps(dict(probe_exit_code=status['exit_code'],archive=record,independent_audit_passed=code==0,full_goal_met=False),indent=2))
    finally:(out/'exit.json').write_text(json.dumps(dict(exit_code=code,wall_seconds=time.monotonic()-begin)))
    sys.exit(code)
