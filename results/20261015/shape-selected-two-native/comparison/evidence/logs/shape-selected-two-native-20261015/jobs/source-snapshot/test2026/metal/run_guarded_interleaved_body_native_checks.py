"""Original 28 gates plus native large-body layout CTest; preserve baseline."""
from pathlib import Path
wrapper=Path(__file__).with_name('run_guarded_wide_batch_prefill_checks.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-wide-batch-prefill','build-interleaved-bf16-body')
source=source.replace("'METAL_QWEN35_WIDE_BATCH_PREFILL'","'METAL_QWEN35_WIDE_BATCH_PREFILL','METAL_INTERLEAVED_BF16_BODY'")
exec(compile(source,__file__,'exec'))
