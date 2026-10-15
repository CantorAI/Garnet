"""Original full native gates followed by B2 chunk80 first/last trained references."""
from pathlib import Path
wrapper=Path(__file__).with_name('run_guarded_interleaved_head_checks.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-interleaved-screen-index','build-wide-batch-prefill')
source=source.replace("'METAL_INTERLEAVED_SCREEN_INDEX'","'METAL_INTERLEAVED_SCREEN_INDEX','METAL_QWEN35_WIDE_BATCH_PREFILL'")
source+="\nrun('wide-batch-trained-reference',[sys.executable,str(s/'test_wide_batch_prefill_reference.py'),runtime,str(build),str(r/'source'),str(r/'models/Qwen3.5-0.8B'),str(output/'wide-reference')])\nassert len(statuses)==28\n"
exec(compile(source,__file__,'exec'))
