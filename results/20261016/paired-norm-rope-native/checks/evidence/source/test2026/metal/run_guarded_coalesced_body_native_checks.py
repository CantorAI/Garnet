"""All original28 stages plus native exact block storage primitive coverage."""
from pathlib import Path
wrapper=Path(__file__).with_name('run_guarded_interleaved_body_native_checks.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('build-interleaved-bf16-body','build-coalesced-bf16-body')
source=source.replace("'METAL_INTERLEAVED_BF16_BODY'","'METAL_INTERLEAVED_BF16_BODY','METAL_COALESCED_BF16_BODY'")
exec(compile(source,__file__,'exec'))
