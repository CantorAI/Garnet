"""Archive terminal wide-batch gates, complete native outputs and exact identities."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_interleaved_head.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('build-interleaved-screen-index','build-wide-batch-prefill'),('build-small-screen-seed','build-interleaved-screen-index'),('interleaved-head-checks-20261015','wide-batch-prefill-checks-20261015'),('interleaved-head-native-20261015','wide-batch-prefill-native-20261015'),('interleaved-head-staging','wide-batch-prefill-staging'),('interleaved-head-original-source-20261015','wide-batch-prefill-original-source-20261015'),('d27e746573098a1668c06a781d53396b65c6ffca87e1f4cf19e86cc5d326c876','837d6f3a55598b3169be7569a00a2ab5935fd60e4643546131790a2e654e4471')]:source=source.replace(old,new)
source=source.replace("'.metal','.executed')","'.metal','.executed','.cmake')")
needle='proof=dict(files={}';assert source.count(needle)==1
source=source.replace(needle,'''import subprocess
assert all(v['exit_code']==0 for v in jobs.values())
closure=json.loads((root/'logs'/native_trial/'source-identities.json').read_text())
for name,sha in closure.items():
    p=root/'source'/name;assert digest(p)==sha;entries[str(p.relative_to(root))]=p
for model in ('Qwen3-0.6B','Qwen2.5-Coder-0.5B-Instruct','Qwen3.5-0.8B'):
    for name in ('model.safetensors','tokenizer.json'):
        p=root/'models'/model/name;omitted[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=digest(p))
p=Path('/Users/zsdx/CantorAI/out/build/macos-arm64-release/bin/xlang3');omitted['external/xlang3']=dict(bytes=p.stat().st_size,sha256=digest(p))
for name in ('garnet_wide_batch_prefill_test','libgarnet.dylib'):
    p=root/'build-wide-batch-prefill'/name;omitted[str(p.relative_to(root))]=dict(bytes=p.stat().st_size,sha256=digest(p))
hardware={name:subprocess.check_output(['/usr/sbin/sysctl','-n',name],text=True).strip() for name in ('machdep.cpu.brand_string','hw.memsize')}
hardware['macos']=subprocess.check_output(['/usr/bin/sw_vers'],text=True)
'''+needle)
source=source.replace('proof=dict(files={},','proof=dict(hardware=hardware,files={},')
exec(compile(source,__file__,'exec'))
