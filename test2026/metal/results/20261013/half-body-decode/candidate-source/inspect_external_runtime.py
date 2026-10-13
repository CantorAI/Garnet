import json,subprocess
pid=65663;chain=[]
for step in range(5):
 result=subprocess.run(['/bin/ps','-p',str(pid),'-o','ppid=,comm='],capture_output=True,text=True)
 if result.returncode or not result.stdout.strip():break
 parts=result.stdout.strip().split(None,1);chain.append(dict(pid=pid,parent=int(parts[0]),name=parts[1]));pid=int(parts[0])
 if pid<=1:break
libraries=subprocess.run(['/usr/sbin/lsof','-p','65663','-Fn'],capture_output=True,text=True)
markers=[name for name in ('libgarnet','libmlx','libtorch','metal.framework','metalperformanceshaders.framework','libopenvino') if name in libraries.stdout.lower()]
print(json.dumps(dict(observed_process=65663,live=bool(chain),ancestry_names=chain,inference_library_markers=markers)))
