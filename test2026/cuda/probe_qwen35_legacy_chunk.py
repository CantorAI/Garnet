"""Run unchanged chunk checks against the archived pre-workspace-API DLL.

Diagnostic only: omit the new Python setter when loading that historical DLL.
This does not alter a fixture, expected values, graph math or test tolerances.
Use only the small capacity32 fixture, whose baseline build fits 64MiB.
"""
import inspect,os,sys,textwrap
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import xModel.qwen35.vl_0_8b.cuda_decoder as module
source=textwrap.dedent(inspect.getsource(module.CUDADecoder._compile))
line="    if kind=='full_attention' and self.batch==8:graph.set_builder_workspace_bytes(128<<20)\n"
assert source.count(line)==1,'Legacy adapter must omit exactly the workspace setter'
namespace=dict(vars(module));exec(source.replace(line,''),namespace)
module.CUDADecoder._compile=namespace['_compile']
path=os.path.join(os.path.dirname(os.path.abspath(__file__)),'test_qwen35_chunk.py')
with open(sys.argv[1]) as stream:
    import json
    assert json.load(stream)['capacity']==32,'Legacy workspace bypass is restricted to capacity32 diagnostics'
with open(path) as stream:code=stream.read()
exec(compile(code,path,'exec'),{'__name__':'__main__','__file__':path})
