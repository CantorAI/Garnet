from pathlib import Path
import hashlib,json,subprocess,sys
w=Path(__file__).resolve().parent;p=w.parent
subprocess.run([sys.executable,str(w/'prepare.py')],check=True)
try:
    subprocess.run([sys.executable,str(w/'controller.py'),'control-full-1','candidate-full-1','candidate-full-2','control-full-2'],check=True)
    subprocess.run([sys.executable,str(p/'dependency_closure.py'),'production'],check=True)
    repo=Path('/Users/shawn/CantorAI/Garnet');source=Path('/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source')
    for root,name in [(repo,'canonical-source-before-build.json'),(source,'control-source-before-dispatch.json')]:
        assert all(hashlib.sha256((root/rel).read_bytes()).hexdigest()==digest for rel,digest in json.loads((p/name).read_text()).items())
    for i in [1,2]:subprocess.run([sys.executable,str(w/'analyze.py'),str(w),'control-full-'+str(i),'candidate-full-'+str(i)],check=True)
finally:
    subprocess.run([sys.executable,str(p/'restore_normal.py')],check=True)
