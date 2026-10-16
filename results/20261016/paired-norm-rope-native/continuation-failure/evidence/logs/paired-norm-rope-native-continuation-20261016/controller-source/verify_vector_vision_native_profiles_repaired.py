"""Repair only the final expanded-script pathname assertion in the frozen full audit."""
from pathlib import Path
import hashlib
p=Path(__file__).with_name('verify_vector_vision_native_profiles.py')
assert hashlib.sha256(p.read_bytes()).hexdigest()=='e53bbdd3bd4449435811aa6924ba0eb4c03b3fc7832c0296c69ffda45d059071'
source=p.read_text()
old="assert (native/(folder.name+'.executed.py')).read_bytes()==expanded"
new="assert (native/folder.with_suffix('.executed.py').name).read_bytes()==expanded"
assert source.count(old)==1;source=source.replace(old,new)
old="result['regenerated_native_comparators']=6"
assert source.count(old)==1
source=source.replace(old,old+"\nresult['expanded_script_paths']={Path(p['evidence']).name:Path(p['evidence']).with_suffix('.executed.py').name for p in profiles}\nresult['distinct_expanded_script_files']=len(set(result['expanded_script_paths'].values()))\nassert result['distinct_expanded_script_files']==3\nresult['audit_path_repair_only']=True")
exec(compile(source,__file__,'exec'))
