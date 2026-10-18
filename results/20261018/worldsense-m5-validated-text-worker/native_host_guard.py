"""Native no-model sidecar guard; never imports model/image implementations."""
import sys,json,os,time
from pathlib import Path
w=Path(__file__).resolve().parent;mode=sys.argv[1];ready=Path(sys.argv[2]);nonce=sys.argv[3];parent=int(sys.argv[4])
hostroot=w/'native-host';legacy=Path('/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-cache')
sys.path.insert(0,str(hostroot));sys.path.insert(0,str(legacy))
import garnet_worker_host
rejected=False
try:garnet_worker_host.legacy_json()
except Exception:rejected=True
assert rejected,'Identity query must reject an unloaded backend'
import garnet
identity=json.loads(garnet_worker_host.legacy_json());support=json.loads(garnet_worker_host.support_json())
assert identity['library_path']==str(legacy/'libgarnet.dylib')and identity['library_sha256']=='dc3d135d664cde077a4333bfeea47252ff83ac64061c3fea562fdac6ccfdac72'
assert identity['loaded_garnet_images']==1 and support['pid']==os.getpid()and support['parent_pid']==parent
assert Path(support['library_path']).resolve()==hostroot/'libgarnet_worker_host.dylib'
raw=json.loads(garnet_worker_host.raw_primary_stats_json());assert len(raw)==11 and all(isinstance(v,int)and not isinstance(v,bool)and v==0 for v in raw.values())
policies=json.loads(garnet_worker_host.raw_policies_json());assert policies['lazy_default_planned_slots']==policies['lazy_default_retired_bytes']==policies['lazy_default_rehydrated_bytes']==0 and 'Structural zeros'in policies['counter_origin']
clocks=[]
for i in range(10):
 before=time.monotonic();value=garnet_worker_host.monotonic();after=time.monotonic();assert before<=value<=after;clocks.append([before,value,after])
memory=json.loads(garnet_worker_host.memory_stats_json());assert memory['task_info_status']==0 and memory['metal_registry_id']>0
for key in ['metal_allocated_bytes','task_physical_footprint_bytes','task_resident_bytes','task_compressed_bytes','task_internal_bytes','task_virtual_bytes']:
 assert isinstance(memory[key],int)and not isinstance(memory[key],bool)and memory[key]>=0
rejected=False
try:garnet_worker_host.watch_parent(parent+100000)
except Exception:rejected=True
assert rejected and garnet_worker_host.watch_parent(parent)and garnet_worker_host.watch_parent(parent)
support=json.loads(garnet_worker_host.support_json());assert support['watchdog_active']and support['watch_parent_pid']==parent
result=dict(pid=os.getpid(),parent_pid=parent,nonce=nonce,mode=mode,support=support,legacy=identity,memory=memory,raw_primary=raw,raw_policy=policies,clock_containment=clocks,unloaded_backend_rejected=True,wrong_parent_rejected=True,model_calls=0,scope='Native C++ metadata/identity/clock/Mach/Metal/watchdog only. No model loading/inference, GPU kernel dispatch, worker routing or performance admission.')
ready.write_text(json.dumps(result,indent=2)+'\n')
if mode=='watch':time.sleep(30)
else:print('Native host identity/clock/actual memory/RAW logical domains/ownership guards PASS; no model inference',flush=True)
