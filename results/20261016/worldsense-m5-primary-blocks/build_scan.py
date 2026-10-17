from pathlib import Path
import shutil,subprocess,hashlib,json,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI/out/research/worldsense-primary-blocks');s=r/'cpu-source';p=s/'scan.cpp'
shutil.copy2(p,w/'pre-execution-scan.cpp')
t=p.read_text().replace('#include <map>','#include <map>\n#include <memory>').replace('Need(ptr->Open(checkpoint,e),e.c_str());','if(!ptr->Open(checkpoint,e))throw std::runtime_error(e);').replace('Need(file.Open(entry->filePath,error),error.c_str());','if(!file.Open(entry->filePath,error))throw std::runtime_error(error);');p.write_text(t)
compiler=subprocess.check_output(['xcrun','--find','clang++'],text=True).strip();args=[compiler,'-std=c++17','-O3','-Wno-deprecated-declarations','-I',str(s)];commands=[]
with(w/'final-build.log').open('w')as log:
 for name in ['scan','safetensors_index']:
  cmd=args+['-MD','-MF',str(r/(name+'.d')),'-c',str(s/(name+'.cpp')),'-o',str(r/(name+'.o'))];commands.append(cmd);subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 cmd=[compiler,str(r/'scan.o'),str(r/'safetensors_index.o'),'-o',str(r/'scan')];commands.append(cmd);subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
dep=[]
for p in r.glob('*.d'):
 text=p.read_text().replace('\\\n',' ');dep+=text.split(':',1)[1].split()
dep+=list(map(str,s.rglob('*.h')))+list(map(str,s.rglob('*.cpp')))+[compiler]
files={str(Path(p)):hashlib.sha256(Path(p).read_bytes()).hexdigest()for p in sorted(set(dep))}
(w/'execution-freeze.json').write_text(json.dumps({'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':commands,'compiler':subprocess.check_output([compiler,'--version'],text=True),'os':subprocess.check_output(['/usr/bin/sw_vers'],text=True),'sdk':subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip(),'dependency_closure_sha256':files,'executable_sha256':hashlib.sha256((r/'scan').read_bytes()).hexdigest(),'selection_sha256':hashlib.sha256((w/'matrix-selection.json').read_bytes()).hexdigest(),'pre_execution_review':'First build succeeded but Open failure-message argument lifetime depended on unspecified argument order. Fixed before any execution; initial source and build log preserved. No GPU or inference run.'},indent=2))
print('Final native CPU source/executable/compiler dependency closure frozen',len(files),flush=True)
