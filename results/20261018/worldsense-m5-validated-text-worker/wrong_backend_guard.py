"""Identical bytes at an unpinned path must not gain native counter admission."""
import sys,json
from pathlib import Path
w=Path(__file__).resolve().parent
sys.path.insert(0,str(w/'native-host'));sys.path.insert(0,str(w/'wrong-backend'))
import garnet_worker_host,garnet
errors=[]
for name in ['legacy_json','memory_stats_json','raw_primary_stats_json','raw_policies_json']:
 rejected=False
 try:getattr(garnet_worker_host,name)()
 except Exception as error:rejected=True;errors.append(dict(api=name,error=str(error)))
 assert rejected,'Unpinned backend admitted native statistics: '+name
(w/'wrong-backend-result.json').write_text(json.dumps(dict(all_four_identity_memory_logical_APIs_reject_unpinned_loaded_backend=True,errors=errors,no_model_inference=True),indent=2)+'\n')
print('Unpinned loaded backend rejected for identity/actual memory/structural counters; no model inference',flush=True)
