"""Run only with native XLang3. No model/image import, preparation or inference."""
import sys,json,os
from pathlib import Path
w=Path(__file__).resolve().parent
source=Path('/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source')
build=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-cache')
for p in [source,source/'runtime',build,w/'mock-adapter']:sys.path.insert(0,str(p))
import garnet,metal_serving
assert Path(metal_serving.__file__).resolve()==(w/'mock-adapter/metal_serving.py').resolve()
status=json.loads(garnet.serve_status_json('ws61-missing'))
(w/'probe-status-before-assert.json').write_text(json.dumps(status,indent=2)+'\n')
assert status['probe_marker']=='ws61-isolated-native-adapter'and not status['ready']
assert Path(status['adapter_source']).resolve()==(w/'mock-adapter/metal_serving.py').resolve()
result=json.loads(garnet.infer_json('no-model-probe','',1,'ws61-missing'))
(w/'probe-rejection-before-assert.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['status']=='error'and result['probe_marker']=='ws61-isolated-native-adapter'
assert result['calls']==dict(status=1,infer_rejection=1,model_loads=0)
(w/'native-module-probe-result.json').write_text(json.dumps(dict(pid=os.getpid(),backend=garnet.backend_info(),adapter_source=str(w/'mock-adapter/metal_serving.py'),native_package_status_and_error_callbacks_reach_preimported_adapter=True,checkpoint_or_tokenizer_or_image_or_actual_model_calls=0,source_metal_serving_not_mutated=True,scope='Fresh native XLang3 + exact validated WS10 Garnet DLL with an isolated mock callback module. No model/GPU inference, throughput, correctness or worker-lifecycle admission.',no_speed_claim=True),indent=2)+'\n')
print('Validated WS10 native package dispatches status/no-model rejection to isolated preimported adapter; no model inference',flush=True)
