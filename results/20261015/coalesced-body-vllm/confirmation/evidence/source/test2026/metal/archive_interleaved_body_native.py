"""Preserve terminal full native body gates, unchanged controls and complete outputs."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_wide_batch_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('build-wide-batch-prefill','build-interleaved-bf16-body'),('build-interleaved-screen-index','build-wide-batch-prefill'),('wide-batch-prefill-checks-20261015','interleaved-body-native-checks-20261015'),('wide-batch-prefill-native-20261015','interleaved-body-native-20261015'),('wide-batch-prefill-staging','interleaved-body-native-staging'),('wide-batch-prefill-original-source-20261015','interleaved-body-native-original-source-20261015')]:source=source.replace(old,new)
needle="assert libraries['build-wide-batch-prefill']=='837d6f3a55598b3169be7569a00a2ab5935fd60e4643546131790a2e654e4471'";assert source.count(needle)==1
source=source.replace(needle,"assert libraries['build-wide-batch-prefill']==json.loads((root/'work/wide-batch-next-comparator-selection.json').read_text())['builds']['wide_batch']['sha256']")
source=source.replace('proof=dict(hardware=hardware,',"assert libraries['build-wide-batch-prefill']=='9ee337d9db7d14bcff29a90970c4b5ac9021aee9f89a38a7f6f364051927d77b'\nproof=dict(hardware=hardware,")
source=source.replace("('garnet_wide_batch_prefill_test','libgarnet.dylib')","('garnet_wide_batch_prefill_test','garnet_interleaved_bf16_body_test','libgarnet.dylib')")
exec(compile(source,__file__,'exec'))
