"""CPU supervisor: raw correctness ledger, exploratory native pilot and NVML evidence."""
import argparse,hashlib,json,os,uuid
from pathlib import Path
from metrics import summarize
from run_logged import run,source_identity
from sample_vram import Sampler

def main():
    p=argparse.ArgumentParser();p.add_argument('--log-root',type=Path,required=True);p.add_argument('--pilot',type=Path,required=True)
    p.add_argument('--artifact',action='append',default=[]);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('Native correctness command required')
    root=a.log_root;root.mkdir(parents=True,exist_ok=True);key=uuid.uuid4().hex
    os.environ['GARNET_STUDY_RUN_ID']=key
    report={'run_id':key,'source':source_identity(),'status':'running','command':command,
        'scope':'Correctness plus exploratory direct native calls; excludes HTTP, not a matched2x comparison'}
    path=root/(key+'-manifest.json');path.write_text(json.dumps(report,indent=2))
    sampler=None;code=1
    try:
        sampler=Sampler(root/(key+'-vram.jsonl'));sampler.phase='whole-build-and-correctness-job'
        report['gpu']=sampler.device;sampler.start()
        code=run(command,root,'correctness','full-vlm-and-separate-native-pilot',a.artifact)
        report['exit_code']=code;report['status']='completed' if code==0 else 'failed'
    except Exception as error:
        report['status']='failed';report['supervisor_error']=repr(error);code=1
    finally:
        if sampler is not None:
            try:
                sampler.close();report['whole_job_memory']=sampler.peak('whole-build-and-correctness-job')
                report['whole_job_memory']['measurement_window']='Entire build plus correctness/pilot process; includes transient builder allocations; not warmed serving peak'
            except Exception as error:
                report['status']='failed';report['sampler_error']=repr(error);code=1
            report['sampling_errors']=sampler.errors
        runtime=root.parent.parent/'build/bin'
        report['runtime_binary_sha256']={name:hashlib.sha256((runtime/name).read_bytes()).hexdigest() for name in ['garnet.dll','xlang3.exe']}
        if a.pilot.exists():
            raw=a.pilot.read_bytes();pilot=json.loads(raw)
            if pilot.get('run_id')==key and pilot.get('status')=='completed':
                rows=[row for row in pilot['rows'] if row['measured']]
                if len(rows)==8 and all(r['generated_tokens']==32 for r in rows):
                    report['exploratory_direct_call_summary']=summarize(rows)
                    report['exploratory_direct_call_summary']['scope']=pilot['scope']
                else:report['pilot_profile_error']='Eight measured32-token requests required'
                report['pilot_sha256']=hashlib.sha256(raw).hexdigest()
        report['source_after']=source_identity();path.write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
