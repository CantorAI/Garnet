"""Freeze terminal native block gates, original sources and complete serving outputs."""
from pathlib import Path
wrapper=Path(__file__).with_name('archive_interleaved_body_native.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
for old,new in [('build-interleaved-bf16-body','build-coalesced-bf16-body'),('interleaved-body-native-checks-20261015','coalesced-body-native-checks-20261015'),('interleaved-body-native-20261015','coalesced-body-native-20261015'),('interleaved-body-native-staging','coalesced-body-native-staging'),('interleaved-body-native-original-source-20261015','coalesced-body-native-original-source-20261015')]:source=source.replace(old,new)
source=source.replace("'garnet_interleaved_bf16_body_test','libgarnet.dylib'","'garnet_interleaved_bf16_body_test','garnet_coalesced_bf16_body_test','libgarnet.dylib'")
needle='proof=dict(hardware=hardware,';assert source.count(needle)==1
source=source.replace(needle,"off=root/'logs/coalesced-native-disabled-compile-20261015'\noff_report=json.loads((off/'report.json').read_text());assert all(v['exit_code']==0 for v in off_report['cases'])\nassert off_report['source_sha256']==digest(root/'source/src/backends/metal/metal_context.mm')\nfor p in off.glob('*'):\n    if p.is_file():entries[str(p.relative_to(root))]=p\n"+needle)
source=source.replace(needle,"coverage=root/'logs/coalesced-checkpoint-coverage-20261015'\ncoverage_report=json.loads((coverage/'report.json').read_text());assert coverage_report['source_sha256']==digest(coverage/'source.py')\nfor p in coverage.glob('*'):\n    if p.is_file():entries[str(p.relative_to(root))]=p\n"+needle)
exec(compile(source,__file__,'exec'))
