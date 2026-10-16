"""Archive terminal six-profile screened-head comparison and actual dtype audits."""
from pathlib import Path
source=Path(__file__).with_name('archive_global_alpha_vllm.py').read_text().replace("glob('global-alpha-vllm-20261014*')","glob('screened-head-vllm-20261015*')").replace('work/global-alpha-next-comparator-selection.json','work/screened-head-next-comparator-selection.json')
needle="for name in names:files[";assert source.count(needle)==1
source=source.replace(needle,"names=names+('compare_screened_head_vllm.py','run_screened_head_vllm.py','launch_screened_head_vllm.py','archive_screened_head_vllm.py','run_threaded_striped_frame_process.py','run_webp_thread_frame_process.py','frame_threaded_striped.py','frame_webp_threaded.py','frame_preprocessing_striped.py')\n"+needle)
needle='manifest={}';assert source.count(needle)==1
source=source.replace(needle,"for name,sha in selection['source_manifest'].items():\n    p=r/'source'/name;assert hashlib.sha256(p.read_bytes()).hexdigest()==sha;files['evidence/source/'+name]=p\nfor p in (r/'source/xModel').rglob('*.py'):files['evidence/'+str(p.relative_to(r))]=p\n"+needle)
exec(compile(source,__file__,'exec'))
