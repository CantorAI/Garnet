"""Archive/audit native comparison, then sequentially test BF16 graph storage."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,time
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
if len(sys.argv)==3:
    out.mkdir(parents=True,exist_ok=False)
    with (out/'output.log').open('w') as stream:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),str(root),str(out),'--worker'],
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    (out/'pid.json').write_text(json.dumps(dict(pid=p.pid)));print(json.dumps(dict(pid=p.pid,output=str(out))))
else:
    assert sys.argv[3]=='--worker'
    start=time.monotonic();code=1;state='waiting_validation_controller';commands=[]
    helpers=Path(__file__).parent;parent=Path(json.loads((root/'preparation.json').read_text())['parent'])
    def wait(folder,expected=None):
        handle=json.loads((folder/'pid.json').read_text());pid=handle['pid']
        if expected is not None:assert pid==expected
        while not (folder/'exit.json').exists():
            live=subprocess.run(['ps','-p',str(pid),'-o','pid=,state=,etime=,comm='],capture_output=True,text=True)
            (out/'last-handle-observation.json').write_text(json.dumps(dict(pid=pid,folder=str(folder),
                exit_code=live.returncode,observation=live.stdout,observed_unix=time.time()),indent=2))
            if live.returncode!=0:
                time.sleep(1)
                assert (folder/'exit.json').exists(),'Known handle missing without terminal record; do not restart'
            time.sleep(2)
        return json.loads((folder/'exit.json').read_text())
    def run(label,command):
        with (out/(label+'.log')).open('w') as stream:result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT)
        commands.append(dict(label=label,command=command,exit_code=result.returncode))
        (out/'commands.json').write_text(json.dumps(commands,indent=2));assert result.returncode==0,(label,result.returncode)
    def freeze_audit(label,owner,archive,archiver,auditor,audit_result):
        run(label+'-archive',[sys.executable,str(helpers/archiver),str(owner),str(archive)])
        with archive.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        audit=owner/'validation-audits'/archive.name.removesuffix('.tar.gz')
        run(label+'-extract',[sys.executable,str(parent/'work/freeze_mlp_evidence.py'),str(archive),str(audit),sha])
        command=[sys.executable,str(helpers/auditor),str(audit),str(audit_result)]
        if label=='bf16':command.append(str(archive))
        run(label+'-audit',command)
        shutil.copyfile(audit_result,out/(label+'-verification.json'))
    try:
        status=wait(root/'logs/shape-selected-two-finalize-20261015',47970)
        assert status['exit_code']==0 and status['state']=='native_benchmark_dispatched'
        state='waiting_exact_native'
        native=root/'logs/shape-selected-two-native-20261015';status=wait(native)
        state='freezing_native'
        archive=root/'shape-selected-two-native-20261015.tar.gz'
        # The archiver accepts a terminal failure. Preserve it before rejecting admission.
        run('native-archive',[sys.executable,str(helpers/'archive_shape_selected_two_profiles.py'),str(root),str(archive)])
        assert status['exit_code']==0,'Native failure frozen; no BF16 probe admitted'
        with archive.open('rb') as stream:sha=hashlib.file_digest(stream,'sha256').hexdigest()
        audit=root/'validation-audits/shape-selected-two-native-20261015'
        run('native-extract',[sys.executable,str(parent/'work/freeze_mlp_evidence.py'),str(archive),str(audit),sha])
        result=root/'shape-selected-two-native-verification.json'
        run('native-audit',[sys.executable,str(helpers/'verify_shape_selected_two_profiles.py'),str(audit),str(result)])
        shutil.copyfile(result,out/'native-verification.json')
        state='launching_bf16_primitive'
        primitive=parent/'logs/mlp-bf16-feed-primitive-20261015'
        run('launch-bf16',[sys.executable,str(helpers/'launch_mlp_bf16_feed_primitive.py'),str(parent),str(primitive),'primitive'])
        state='waiting_exact_bf16_primitive';status=wait(primitive)
        archive=parent/'mlp-bf16-feed-primitive-20261015.tar.gz'
        if status['exit_code']!=0:
            run('bf16-archive',[sys.executable,str(helpers/'archive_mlp_bf16_feed_primitive.py'),str(parent),str(archive)])
            raise RuntimeError('BF16 primitive failed; terminal failure archived')
        state='auditing_bf16_primitive'
        freeze_audit('bf16',parent,archive,'archive_mlp_bf16_feed_primitive.py','verify_mlp_bf16_feed_primitive.py',parent/'mlp-bf16-feed-primitive-verification.json')
        state='native_and_bf16_primitive_audited';code=0
    finally:
        (out/'exit.json').write_text(json.dumps(dict(exit_code=code,state=state,wall_seconds=time.monotonic()-start)))
    sys.exit(code)
