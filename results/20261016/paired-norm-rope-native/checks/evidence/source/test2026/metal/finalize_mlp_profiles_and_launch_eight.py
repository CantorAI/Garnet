"""After exact native comparison: archive, audit full outputs, then compact decode."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tarfile,time

root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
assert len(sys.argv)==3 or (len(sys.argv)==4 and sys.argv[3]=='--worker')
upstream=root/'logs/mlp-prefill-native-finalize-20261015';upstream_pid=82980
assert json.loads((upstream/'pid.json').read_text())['pid']==upstream_pid
if len(sys.argv)==3:
    assert not (root/'logs/coalesced-eight-body-20261015').exists()
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=process.pid)))
    print(json.dumps(dict(pid=process.pid,output=str(out))))
else:
    started=time.monotonic();code=1;state='waiting_for_native_dispatch';steps=[]
    def wait_terminal(folder,pid,script):
        begin=time.monotonic()
        assert json.loads((folder/'pid.json').read_text())['pid']==pid
        while not (folder/'exit.json').exists():
            os.kill(pid,0)
            args=subprocess.check_output(['/bin/ps','-p',str(pid),'-o','args='],text=True)
            assert script in args and str(folder) in args and '--worker' in args
            if time.monotonic()-begin>7200:raise TimeoutError('Exact dependency still lacks terminal evidence; no restart')
            time.sleep(5)
        return json.loads((folder/'exit.json').read_text())
    def run(name,command):
        print('Starting',name,flush=True)
        with (out/(name+'.log')).open('w') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        steps.append(dict(name=name,command=command,exit_code=result.returncode))
        (out/'steps.json').write_text(json.dumps(steps,indent=2)+'\n')
        assert result.returncode==0,name
        print('Completed',name,flush=True)
    try:
        status=wait_terminal(upstream,upstream_pid,'finalize_mlp_checks_and_launch_profiles.py')
        assert status['exit_code']==0 and status['state']=='native_benchmark_dispatched'
        dispatch=json.loads((upstream/'benchmark-dispatch.json').read_text())
        native=root/'logs/mlp-prefill-native-20261015'
        state='waiting_for_exact_native_comparison'
        status=wait_terminal(native,dispatch['pid'],'launch_mlp_prefill_native_profiles.py')
        state='archiving_native_comparison';scripts=root/'source/test2026/metal'
        archive=root/'mlp-prefill-native-20261015.tar.gz';assert not archive.exists()
        run('archive-native',[sys.executable,str(scripts/'archive_mlp_prefill_native_profiles.py'),str(root),str(archive)])
        assert status['exit_code']==0,'Native comparison failed; preserve archive and do not launch GPU probe'
        state='auditing_complete_outputs';audit=root/'validation-audits/mlp-prefill-native-20261015'
        audit.mkdir(parents=True,exist_ok=False)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                assert member.isfile();p=(audit/member.name).resolve();assert p.is_relative_to(audit)
                p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(tar.extractfile(member).read())
        verification=root/'mlp-prefill-native-verification.json';assert not verification.exists()
        run('audit-native',[sys.executable,str(scripts/'verify_mlp_prefill_native_comparison.py'),str(audit),str(verification)])
        checked=json.loads(verification.read_text());assert checked['full_output_requests']==72 and checked['actual_graph_routes']
        with archive.open('rb') as stream:checked['archive_sha256']=hashlib.file_digest(stream,'sha256').hexdigest()
        verification.write_text(json.dumps(checked,indent=2)+'\n')
        state='dispatching_compact_eight_probe';probe=root/'logs/coalesced-eight-body-20261015'
        run('launch-eight-probe',[sys.executable,str(scripts/'launch_coalesced_eight_body.py'),str(root),str(probe),'component'])
        dispatched=json.loads((probe/'pid.json').read_text());os.kill(dispatched['pid'],0)
        (out/'component-dispatch.json').write_text(json.dumps(dict(**dispatched,output=str(probe),native_archive_sha256=checked['archive_sha256']),indent=2)+'\n')
        state='eight_component_dispatched';code=0
    finally:
        (out/'exit.json').write_text(json.dumps(dict(exit_code=code,state=state,wall_seconds=time.monotonic()-started,
            scope='Native archival/audit and independent decode component dispatch; no promotion, repeat or vLLM goal-completion claim'))+'\n')
    sys.exit(code)
