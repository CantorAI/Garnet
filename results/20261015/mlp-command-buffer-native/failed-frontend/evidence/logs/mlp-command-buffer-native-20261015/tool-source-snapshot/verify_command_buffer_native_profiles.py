"""Audit all72 native outputs, routes, clocks and both immutable source closures."""
from pathlib import Path
import hashlib,json,re,statistics,sys

base=Path(sys.argv[1]).resolve();root=base/'evidence'
proof=json.loads((root/'manifest.json').read_text())
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
for name,info in proof['files'].items():
    path=(base/name).resolve();assert path.is_relative_to(base)
    assert path.stat().st_size==info['bytes'] and digest(path)==info['sha256']
native_trial='mlp-command-buffer-native-20261015'
assert proof['jobs'][native_trial]['exit_code']==0
saved=sys.argv[:]
try:
    sys.argv=[saved[0],str(base),str(base/'checks-reverification.json')]
    helper=Path(__file__).with_name('verify_command_buffer_native_checks.py')
    exec(compile(helper.read_text(),str(helper),'exec'),{'__file__':str(helper)})
finally:sys.argv=saved
checks_audit=json.loads((base/'checks-reverification.json').read_text())
audit=json.loads((root/'logs'/native_trial/'checks-verification.json').read_text())
assert audit['archive_sha256']==proof['checks_archive_sha256']
for key,value in checks_audit.items():
    if key=='verified_files':assert value>=audit[key]
    else:assert value==audit[key],key
identities=json.loads((root/'logs'/native_trial/'source-identities.json').read_text())
assert identities==checks_audit['source_manifest']
for name,sha in identities.items():
    assert digest(root/'source'/name)==sha
    assert digest(root/'logs'/native_trial/'source-snapshot'/name)==sha
external=json.loads((root/'logs'/native_trial/'external-tool-identities.json').read_text())['files']
for name,sha in external.items():
    assert digest(root/'external-tools'/name)==sha
    assert digest(root/'logs'/native_trial/'tool-source-snapshot'/name)==sha
template=root/'source/test2026/metal/verify_mlp_prefill_native_profiles.py'
assert digest(template)=='027b85935a861df5040a4a39128cda09f3fe9e814348eb58c4d5e7198eb6ecf8'
source=template.read_text();source=source[source.index("native=root/'logs'/native_trial/'native'"):]
source=source.replace('build-mlp-prefill-graph-guardfix','build-mlp-command-buffer')
needle="==submissions+expected_graphs";assert source.count(needle)==1
source=source.replace(needle,'==submissions')
needle="queue='same-native-queue',synchronous=True,lazy_graph_compile=True,encode_ms_includes_graph_gpu_work=True"
assert source.count(needle)==1
source=source.replace(needle,"queue='native-MPSCommandBuffer',synchronous=False,lazy_graph_compile=True,refreshes_root=True,final_completion_checked=True")
needle="        assert proof['omitted']['models/'+model+'/'+name]['sha256']==meta['assets'][model][name]"
assert source.count(needle)==1
source=source.replace(needle,"""        asset='models/'+model+'/'+name
        if 'evidence/'+asset in proof['files']:assert digest(root/asset)==meta['assets'][model][name]
        else:assert proof['omitted'][asset]['sha256']==meta['assets'][model][name]""")
needle='source_closure_files=len(identities),actual_graph_routes=True,checks=checks_audit,'
assert source.count(needle)==1
source=source.replace(needle,"source_closure_files=len(identities),external_tool_files=len(external),original_checks_archive_verified_files=audit['verified_files'],actual_graph_routes=True,checks=checks_audit,")
exec(compile(source,__file__,'exec'))
