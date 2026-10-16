import json,sys
from pathlib import Path
root=Path('/Users/zsdx/garnet-metal-investigation/logs')
for name in sys.argv[1:]:
 folder=root/name;print(name)
 if (folder/'exit.json').exists():print('exit',json.loads((folder/'exit.json').read_text())['exit_code'])
 else:print('running')
 for line in (folder/'output.log').read_text().splitlines():
  if line.startswith(('metal_text_result ','metal_image_result ')):
   data=json.loads(line.split(' ',1)[1]);keys=['request','batch','text','finished','initialization_seconds','vision_seconds','prefill_seconds','ttft_loaded_seconds','total_loaded_seconds']
   result={key:data[key] for key in keys if key in data};steps=data['decode_step_seconds'];result['decode_tps']=len(steps)/sum(steps) if steps else None;print(json.dumps(result))
  elif line.startswith(('metal_text_process ','metal_image_process ')):print(line)
