"""Owned native XLang3 parent for watchdog correctness. No speed result."""
import sys,os,time,json
from pathlib import Path
w=Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-native-text-worker');case=sys.argv[1];folder=Path(sys.argv[2]);s=Path('/Users/shawn/CantorAI/out/research/worldsense-native-text-worker/source');b=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-native-text-worker-on');sys.path.insert(0,str(s));sys.path.insert(0,str(b));import garnet
if case=='bootstrap':
 name='invalid-text';model=w/'invalid-text-fixture';cache=w/'invalid-text-cache';profile=dict(maxInputTokens=64,maxOutputTokens=8,textPrefixCache=False)
else:
 name='Qwen3-1.7B';model=Path('/Users/shawn/CantorAI/out/models')/name;cache=Path('/Users/shawn/CantorAI/out/cache')/name;profile=dict(maxInputTokens=512,maxOutputTokens=256,prefillChunk=16,textPrefixCache=False)
status=json.loads(garnet.serve_model(str(model),str(s/'xModel/qwen/metal_dense'),str(cache),json.dumps(profile),name));assert status['ready']==(case!='bootstrap');identity=json.loads(garnet.native_text_worker_info_json(''));assert len(identity['children'])==1
(folder/'ready.json').write_text(json.dumps(dict(parent_pid=os.getpid(),case=case,status=status,identity=identity),indent=2))
while not(folder/'start.request').exists():time.sleep(.01)
if case=='active_infer':
 (folder/'infer-dispatched.json').write_text(json.dumps(dict(parent_pid=os.getpid(),monotonic=time.monotonic())))
 value=json.loads(garnet.infer_json('Write a long, detailed numbered list explaining 100 different kinds of tools and their uses. Continue listing tools with detailed descriptions.','',256,name));(folder/'unexpected-infer-completed.json').write_text(json.dumps(value,indent=2))
while True:time.sleep(.05)
