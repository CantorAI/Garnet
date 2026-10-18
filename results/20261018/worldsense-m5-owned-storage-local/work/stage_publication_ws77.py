"""Stage only the verified experimental source/evidence and five research docs."""
from pathlib import Path
import hashlib,json,subprocess
repo=Path('/Users/shawn/CantorAI/Garnet');prefix='results/20261018/worldsense-m5-owned-storage-local';d=repo/prefix
def git(*args):return subprocess.check_output(['git','-C',str(repo),*args])
assert git('branch','--show-current').decode().strip()=='feature/qwen35-metal'
assert git('rev-parse','HEAD').decode().strip()=='16c46957f528392d44c96b68ff9d0a1d1b1e7b67'
assert not git('diff','--cached','--name-only')
m=json.loads((d/'file-manifest.json').read_text());assert set(m)=={str(p.relative_to(d))for p in d.rglob('*')if p.is_file()and p.name!='file-manifest.json'}
assert all(Path(k).name!='worldsense-http.py'and hashlib.sha256((d/k).read_bytes()).hexdigest()==v for k,v in m.items())
docs=['docs/research/metal_optimization_research_log.md','docs/research/metal-method-evidence/INDEX.md','docs/research/metal-method-evidence/index.csv','docs/research/metal-method-evidence/index.json','docs/research/metal-method-evidence/documentation-check.json']
assert set(git('diff','--name-only','-z').decode().strip('\0').split('\0'))==set(docs)
expected={prefix+'/'+k for k in list(m)+['file-manifest.json']}
assert set(git('ls-files','--others','--exclude-standard','-z').decode().strip('\0').split('\0'))==expected
check=json.loads((repo/docs[-1]).read_text());assert check['status']=='passed'and check['reports_verified']==268
subprocess.run(['git','-C',str(repo),'add','--',prefix,*docs],check=True)
assert set(git('diff','--cached','--name-only','-z').decode().strip('\0').split('\0'))==expected|set(docs)
print('Exactly',len(expected),'verified archive files plus5research docs staged; original production source remains unchanged. Author Shawn <shawn@cantorai.com>.')
