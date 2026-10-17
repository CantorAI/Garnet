from pathlib import Path
import json,hashlib,shutil,subprocess
root=Path('/Users/shawn/Documents/Codex/2026-10-15/i');r=Path('/Users/shawn/CantorAI');w=root/'work/worldsense-wide-cached-prefill';old=root/'work/worldsense-primary-cached-lazy';a=r/'out/research/worldsense-primary-cached-lazy/source';b=r/'out/research/worldsense-wide-primary-lazy/source';s=r/'out/research/worldsense-wide-cached-prefill/source';assert not s.exists();sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();assert sha(w/'native-design-before-implementation.txt')==json.loads((w/'design-identity.json').read_text())['design_sha256']
seed=json.loads((old/'starting-source-manifest.json').read_text());wide=json.loads((root/'work/worldsense-wide-primary-lazy/starting-source-manifest.json').read_text());files=json.loads((root/'work/worldsense-wide-primary-lazy/implementation-audit.json').read_text())['changed_files'];base='f21c1a2584031ee9d1c129d1e58818f9d1c826d2';patch=''
for rel,h in seed.items():assert sha(a/rel)==h,rel
for rel in files:
 assert sha(b/rel)==wide[rel],rel
 original=subprocess.run(['git','-C',str(r/'Garnet'),'show',base+':'+rel],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL)
 if original.returncode:
  patch+='diff --git a/'+rel+' b/'+rel+'\nnew file mode 100644\n--- /dev/null\n+++ b/'+rel+'\n@@ -0,0 +1,'+str(len((b/rel).read_text().splitlines()))+' @@\n'+''.join('+'+line+'\n'for line in(b/rel).read_text().splitlines())
 else:
  import difflib
  patch+=''.join(difflib.unified_diff(original.stdout.decode().splitlines(keepends=True),(b/rel).read_text().splitlines(keepends=True),fromfile='a/'+rel,tofile='b/'+rel,n=1))
for rel in seed:
 q=s/rel;q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(a/rel,q)
(w/'ws52-seed-source-manifest.json').write_text(json.dumps(seed,indent=2)+'\n');(w/'ws46-scope-source-manifest.json').write_text(json.dumps(wide,indent=2)+'\n');(w/'wide46-over-f21.patch').write_text(patch)
p=subprocess.run(['git','apply','--check',str(w/'wide46-over-f21.patch')],cwd=s,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True);(w/'merge-check.log').write_text(p.stdout);print('Patchcheckexit',p.returncode);print(p.stdout)
if p.returncode==0:subprocess.run(['git','apply',str(w/'wide46-over-f21.patch')],cwd=s,check=True);print('Wide46patchapplied')
