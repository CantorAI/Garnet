"""Original complete native ABBA, command-buffer graph versus retained c186."""
from pathlib import Path
import sys

p=Path(__file__).with_name('compare_mlp_prefill_native.py')
wrapper=p.read_text();ns={'__file__':str(p)}
exec(compile(wrapper[:wrapper.rindex('\ndestination=')],str(p),'exec'),ns)
source=ns['source']
def replace(old,new):
    global source
    assert source.count(old)==1,(old,source.count(old))
    source=source.replace(old,new)
replace('build-mlp-prefill-graph-guardfix','build-mlp-command-buffer')
replace('logs/mlp-prefill-native-checks-resume-20261015/jobs/metadata.json',
        'logs/mlp-command-buffer-native-checks-20261015/jobs/metadata.json')
replace("queue='same-native-queue',synchronous=True,lazy_graph_compile=True,encode_ms_includes_graph_gpu_work=True",
        "queue='native-MPSCommandBuffer',synchronous=False,lazy_graph_compile=True,refreshes_root=True,final_completion_checked=True")
replace('OFF-default synchronous same-queue complete MLP graph during prefill.',
        'Private complete MLP graph encoded into the native MPSCommandBuffer, retaining native producers/consumers and final completion checks.')
replace("value['prefill_profile'].get('prefill_partial_submissions',0)==submissions+mlps",
        "value['prefill_profile'].get('prefill_partial_submissions',0)==submissions")
destination=Path(sys.argv[2]).with_suffix('.executed.py')
source=repr(__doc__)+source[source.index('\n'):]
destination.write_bytes(source.encode())
if '--generate-only' not in sys.argv[3:]:exec(compile(source,str(destination),'exec'))
