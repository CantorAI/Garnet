import sys,json,pathlib
sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-inventory')
import garnet
result=json.loads(garnet.memory_inventory_json([],None));assert result['buffers']==[]and result['references']==[]and result['unique_allocated_bytes']==0
checks=[]
for graphs,session in [('wrong',None),([dict(object='wrong',tag='bad')],None),([dict(object=garnet.compiled_graph(),tag='closed')],None)]:
 rejected=False
 try:garnet.memory_inventory_json(graphs,session)
 except Exception:rejected=True
 assert rejected;checks.append(rejected)
assert json.loads(garnet.memory_inventory_json([],None))['unique_allocated_bytes']==0
pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-inventory/api-guard.json').write_text(json.dumps(dict(empty_collection_and_recovery=True,invalid_and_closed_descriptors_rejected=checks,no_addresses_in_empty_schema=True),indent=2))
print('Native XLang3 inventory schema, invalid/closed inputs and recovery PASS')
