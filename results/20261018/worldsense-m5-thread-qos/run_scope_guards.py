from pathlib import Path
import subprocess,json,hashlib,sys
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');s=r/'out/research/worldsense-thread-qos/source';n=r/'out/build/macos-arm64-release/bin';exe=w/'scope-native-test';command=['/usr/bin/c++','-std=c++17','-O2','-I',str(s/'src/entry'),'-I',str(s/'third_party'),str(w/'scope_native_test.cpp'),str(s/'src/entry/native_worker_support.cpp'),'-o',str(exe)]
with(w/'build-scope-native-test.log').open('w')as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
(w/'scope-native-test-build.json').write_text(json.dumps(dict(command=command,source_sha256=hashlib.sha256((w/'scope_native_test.cpp').read_bytes()).hexdigest(),support_sha256=hashlib.sha256((s/'src/entry/native_worker_support.cpp').read_bytes()).hexdigest(),executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest()),indent=2)+'\n')
with(w/'scope-native-test.log').open('w')as log:subprocess.run([str(exe)],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
for mode in ['off','primary','on','fault']:
 with(w/('scope-guard-'+mode+'.log')).open('w')as log:subprocess.run([str(n/'xlang3'),'-B',str(w/'scope_guard.py'),mode],stdout=log,stderr=subprocess.STDOUT,check=True,timeout=30)
 print('Scope native SDK guard PASS '+mode,flush=True)
(w/'scope-guards-result.json').write_text(json.dumps(dict(main_and_stdthread_destructor_unwind_unit_pass=True,nativeSDK_scope_guards_all4roles_pass=True,no_model_or_GPU_inference=True),indent=2)+'\n')
