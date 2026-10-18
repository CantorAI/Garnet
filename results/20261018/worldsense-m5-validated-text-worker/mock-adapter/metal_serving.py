"""No-model dispatch probe; never imports checkpoint/model/image code."""
import json
_calls={'status':0,'infer_rejection':0,'model_loads':0}
def serve_status_json(name):
 _calls['status']+=1
 return json.dumps(dict(ready=False,model_id=name,probe_marker='ws61-isolated-native-adapter',adapter_source=__file__,calls=dict(_calls)))
def infer_json(prompt,image,limit,name):
 _calls['infer_rejection']+=1
 return json.dumps(dict(status='error',model_id=name,error_message='WS61 no-model probe: inference deliberately unavailable',probe_marker='ws61-isolated-native-adapter',adapter_source=__file__,calls=dict(_calls)))
