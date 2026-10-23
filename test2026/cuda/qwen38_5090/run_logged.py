"""Append-only build/correctness command ledger for the study; never a timing benchmark.

Use one ledger writer at a time. Never pass authentication commands or secrets.
Raw output is retained with byte offsets; process duration is not model latency.
"""
import argparse, datetime, hashlib, json, os, subprocess, sys, time, uuid
from pathlib import Path


def source_identity():
    root = Path(__file__).resolve().parents[3]
    def git(*args): return subprocess.check_output(['git','-C',str(root),*args],stderr=subprocess.DEVNULL)
    try:
        head = git('rev-parse','HEAD').decode().strip()
        diff = git('diff','HEAD','--binary')
        names = git('ls-files','--others','--exclude-standard','-z').decode().split('\0')
        untracked = {n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in names if n and (root/n).is_file()}
        return {'commit':head,'tracked_diff_sha256':hashlib.sha256(diff).hexdigest(),'untracked_source_sha256':untracked}
    except (OSError,subprocess.CalledProcessError):
        study = root.parent
        state_file = study/'study-state.json'
        state = json.loads(state_file.read_text()) if state_file.is_file() else {}
        manifests = {}
        for name in ('source-manifest.json','preparation-manifest.json'):
            path = study/name
            if path.is_file(): manifests[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {'commit':state.get('commit'),'source_manifest_sha256':manifests,
                'scope':'Copied source; use the separately verified source/patch manifests'}


def run(command, log_root, kind, label, artifacts=()):
    if kind not in ('build','correctness','setup'): raise ValueError('Benchmark timing requires its separate measurement harness')
    root = Path(log_root)
    root.mkdir(parents=True,exist_ok=True)
    event_id = uuid.uuid4().hex
    def append(event):
        event.update(id=event_id,utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        with (root/'commands.jsonl').open('ab') as f:
            f.write((json.dumps(event,sort_keys=True)+'\n').encode())
    artifact_hashes = {str(p):hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in artifacts}
    append({'event':'started','kind':kind,'label':label,'argv':command,'cwd':str(Path.cwd()),
            'source':source_identity(),'artifact_sha256':artifact_hashes,
            'environment':{k:os.environ[k] for k in ('CUDA_VISIBLE_DEVICES','GARNET_RUNTIME_ROOT') if k in os.environ}})
    start = time.monotonic()
    with (root/'commands.log').open('ab') as output:
        output.write(('\n'+json.dumps({'id':event_id,'kind':kind,'label':label,'argv':command})+'\n').encode())
        offset = output.tell()
        try:
            with subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT) as process:
                while True:
                    block = process.stdout.read1(65536)
                    if not block: break
                    output.write(block); output.flush()
                    sys.stdout.buffer.write(block); sys.stdout.buffer.flush()
                code = process.wait()
        except OSError as error:
            output.write((type(error).__name__+': '+str(error)+'\n').encode())
            code = 127
        end = output.tell()
    append({'event':'finished','kind':kind,'label':label,'exit_code':code,
            'raw_output_file':'commands.log','raw_start_offset':offset,'raw_end_offset':end,
            'process_elapsed_seconds':time.monotonic()-start,
            'scope':'Process execution duration; not TTFT, decode speed or serving throughput',
            'source_after':source_identity()})
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log-root',required=True,type=Path)
    parser.add_argument('--kind',required=True,choices=('build','correctness','setup'))
    parser.add_argument('--label',required=True)
    parser.add_argument('--artifact',action='append',default=[])
    parser.add_argument('command',nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command: parser.error('A build, setup or correctness command is required')
    return run(command,args.log_root,args.kind,args.label,args.artifact)


if __name__ == '__main__': raise SystemExit(main())
