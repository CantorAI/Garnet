"""Isolated CPU-validation option test for streamed original-row MPS bodies."""
from pathlib import Path
wrapper=Path(__file__).with_name('run_streamed_direct_mps_body.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('from make_direct_mps_body import context as generate_context,harness as generate_harness','from make_mps_fastencode_body import context as generate_context,harness as generate_harness')
source=source.replace("paths=[Path(__file__),","paths=[Path(__file__),s/'make_mps_fastencode_body.py',s/'run_streamed_direct_mps_body.py',")
source=source.replace("out/'streamed_direct_mps_body_gpu","out/'fastencode_streamed_mps_body_gpu")
source=source.replace('metadata.update(context_generated_sha256=',"metadata.update(mps_options='SkipAPIValidation only; reduced precision not enabled',context_generated_sha256=")
exec(compile(source,__file__,'exec'))
