"""All six original complete-output native ABBA profiles for selected shapes."""
from pathlib import Path
import hashlib,json,sys
root=Path(sys.argv[1]).resolve();out=Path(sys.argv[2]).resolve()
audit=json.loads((root/'shape-selected-two-checks-verification.json').read_text())
assert audit['validation_stages']==28 and audit['ctests']==7 and audit['coalesced_primitive_cases']==104
for name,sha in audit['source_manifest'].items():
    assert hashlib.sha256((root/'source'/name).read_bytes()).hexdigest()==sha,name
source=Path(__file__).with_name('run_vision_global_alpha_profiles.py').read_text()
source=source.replace("str(s/'compare_vision_global_alpha.py')","str(Path(__file__).with_name('compare_shape_selected_two_native.py'))")
source=source.replace("confirm='--confirm' in sys.argv[3:]",'confirm=False')
needle="worldsense_script=Path('/Users/zsdx/cantorai-release-tests/current/bin/WorldSense/Backend/package_runtime.py')"
assert source.count(needle)==1
source=source.replace(needle,'''import shutil
source_identities=json.loads((r/'shape-selected-two-checks-verification.json').read_text())['source_manifest']
for name,sha in source_identities.items():
    path=r/'source'/name;assert hashlib.sha256(path.read_bytes()).hexdigest()==sha
    copy=output/'source-snapshot'/name;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes(path.read_bytes())
(output/'source-identities.json').write_text(json.dumps(source_identities,indent=2))
tools=Path(__file__).parent
tool_identities={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in tools.iterdir() if p.is_file() and p.suffix in ('.py','.h','.cpp','.mm')}
for name in tool_identities:
    copy=output/'tool-source-snapshot'/name;copy.parent.mkdir(parents=True,exist_ok=True);copy.write_bytes((tools/name).read_bytes())
(output/'external-tool-identities.json').write_text(json.dumps(dict(directory=str(tools),files=tool_identities),indent=2))
shutil.copyfile(r/'shape-selected-two-checks-verification.json',output/'checks-verification.json')
'''+needle)
source+='''
for name,sha in source_identities.items():assert hashlib.sha256((r/'source'/name).read_bytes()).hexdigest()==sha
for name,sha in tool_identities.items():assert hashlib.sha256((tools/name).read_bytes()).hexdigest()==sha
'''
out.parent.mkdir(parents=True,exist_ok=True)
exec(compile(source,__file__,'exec'))
