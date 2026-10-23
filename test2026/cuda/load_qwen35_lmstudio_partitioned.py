"""Use documented server KV fields absent from the installed Python typed SDK.

Only this process adapts the config encoder; installed packages stay unchanged.
Runtime must verify actual non-unified slots/context before timing.
"""
import argparse,json
from pathlib import Path
import lmstudio as lms
import lmstudio.json_api as api
p=argparse.ArgumentParser();p.add_argument('model_key');p.add_argument('identifier');p.add_argument('output')
p.add_argument('--batch',type=int,choices=[1,2,4,8],required=True);a=p.parse_args()
fields=[{'key':'llm.load.numParallelSessions','value':a.batch},
    {'key':'llm.load.useUnifiedKvCache','value':False},
    {'key':'llm.load.speculativeDecoding.draftMtp','value':False}]
original=api.load_config_to_kv_config_stack
def encode(config,config_type):
    stack=original(config,config_type);data=stack.to_dict()
    target=data['layers'][0]['config']['fields'];keys={f['key'] for f in fields}
    data['layers'][0]['config']['fields']=[f for f in target if f['key'] not in keys]+fields
    return type(stack)._from_api_dict(data)
api.load_config_to_kv_config_stack=encode
try:
    with lms.Client() as client:
        assert not client.llm.list_loaded(),'Another model is loaded'
        config={'contextLength':512,'gpu':{'ratio':1.0},'flashAttention':True}
        model=client.llm.load_new_instance(a.model_key,a.identifier,config=config,ttl=None)
        Path(a.output).write_text(json.dumps({'identifier':model.identifier,'requested_config':config,
            'requested_server_fields':fields,'sdk_load_config':model.get_load_config().to_dict(),
            'source':'https://github.com/lmstudio-ai/lmstudio-js/blob/main/packages/lms-kv-config/src/schema.ts',
            'scope':'Requested config only; runtime audit still required'},indent=2))
        print('LM_PARTITIONED_LOAD_COMPLETE',model.identifier,flush=True)
finally:api.load_config_to_kv_config_stack=original
