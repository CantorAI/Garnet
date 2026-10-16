"""Fresh global-alpha build with complete original cases and guarded retry groups."""
import ast,sys
from pathlib import Path
folder=Path(__file__).parent;ns={'__file__':__file__,'__doc__':__doc__}
wrapper=(folder/'run_global_half_alpha_checks.py').read_text();exec(wrapper[:wrapper.index('destination=')],ns)
wrapper=ns['source'];exec(wrapper[:wrapper.index('destination=')],ns);source=ns['source']
tree=ast.parse((folder/'resume_packed_prefill_norm_checks.py').read_text())
retry=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='retry' for t in n.targets))
assert source.count('def run(name,command):')==1
source=source.replace('def run(name,command):','def run_once(name,command):')
needle="assert not build.exists(),'Keep existing build artifacts; use a fresh candidate directory'";assert source.count(needle)==1
source=source.replace(needle,retry+'\n'+needle)
destination=Path(sys.argv[2]).with_suffix('.executed.py');destination.write_bytes(source.encode())
exec(compile(source,str(destination),'exec'))
