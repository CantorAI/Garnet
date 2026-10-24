from pathlib import Path
import hashlib,json,subprocess
repo=Path('D:/CantorAI/work/qwen38-5090');build=repo/'out/qwen38-native';output=Path('C:/Users/shaw9/Documents/Codex/2026-10-23/referenced-chatgpt-conversation-this-is-an/outputs')
cmake='C:/Program Files/Microsoft Visual Studio/2022/Community/Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe';rows=[]
with (output/'vendor-staging-config.log').open('wb') as log:
 for mode in ('OFF','ON'):
  command=[cmake,'-S',str(repo/'test2026/cuda/qwen38_5090'),'-B',str(build),'-DGARNET_STAGE_VENDOR_DLLS='+mode]
  process=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT);log.write(process.stdout);log.flush();assert process.returncode==0
  ninja=(build/'build.ninja').read_text();commands=[line for line in ninja.splitlines() if 'POST_BUILD =' in line and 'registry.json' in line];assert len(commands)==1
  staged='nvinfer_10.dll' in commands[0];assert staged==(mode=='ON')
  rows.append({'mode':mode,'configure_exit':process.returncode,'vendor_post_build_copy':staged,'generated_post_build_command_sha256':hashlib.sha256(commands[0].encode()).hexdigest()})
result={'status':'passed','cases':rows,'scope':'Generated native Windows build commands only. Default staging remains ON; OFF is for the existing study runtime with verified dependencies. No inference or performance claim.'}
(output/'vendor-staging-config-verified.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
