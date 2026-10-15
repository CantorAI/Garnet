"""Archive sequential selected-library comparisons and real comparator dtype audits."""
import hashlib,io,json,platform,subprocess,sys,tarfile
from pathlib import Path
r=Path(sys.argv[1]).resolve();target=Path(sys.argv[2]).resolve();files={}
for folder in sorted((r/'logs').glob('selected-attention-vllm*')):
    assert json.loads((folder/'exit.json').read_text())['exit_code']==0
    summary=json.loads((folder/'jobs/summary.json').read_text());assert len(summary['profiles'])==6 and len(summary['dtype_audits'])==3
    for path in sorted(folder.rglob('*')):
        if path.is_file() and path.suffix in ('.json','.log','.txt','.py'):
            files['evidence/'+str(path.relative_to(r))]=path
names=('archive_selected_attention_vllm.py','compare_selected_attention_vllm.py','run_selected_attention_vllm.py','compare_retained_prefill_vllm.py','run_retained_prefill_vllm_profiles.py','compare_vllm_metal.py','audit_comparator_dtypes.py','run_matched_garnet.py','run_matched_vllm.py','run_matched_frame_process.py','benchmark_matched_process.py','frame_worker_garnet.py','frame_worker_vllm.py','frame_preprocessing.py','background_comparison.py','run_after_owned_job.py')
for name in names:files['evidence/source/test2026/metal/'+name]=r/'source/test2026/metal'/name
files['evidence/selection.json']=r/'work/attention-prefix-next-comparator-selection.json'
selection=json.loads(files['evidence/selection.json'].read_text());libraries={}
for label,item in selection['builds'].items():
    p=r/item['directory']/'libgarnet.dylib';sha=hashlib.file_digest(p.open('rb'),'sha256').hexdigest();assert sha==item['sha256'];libraries[label]=sha
manifest={}
with tarfile.open(target,'w:gz') as archive:
    for name,path in files.items():
        data=path.read_bytes();manifest[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());archive.add(path,arcname=name,recursive=False)
    for name,value in [('evidence/files.json',manifest),('evidence/archive-host.json',dict(libraries=libraries,python=sys.version,platform=platform.platform(),hardware={k:subprocess.check_output(['/usr/sbin/sysctl','-n',k],text=True).strip() for k in ('machdep.cpu.brand_string','hw.memsize','hw.physicalcpu','hw.logicalcpu')}))]:
        data=(json.dumps(value,indent=2)+'\n').encode();info=tarfile.TarInfo(name);info.size=len(data);archive.addfile(info,io.BytesIO(data))
print(json.dumps(dict(files=len(manifest),archive_sha256=hashlib.sha256(target.read_bytes()).hexdigest())))
