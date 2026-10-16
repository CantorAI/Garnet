import sys,json,pathlib
sys.path.insert(0,'/Users/shawn/CantorAI/out/build/garnet-metal-worldsense-prefix-off')
sys.path.insert(0,'/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source')
import garnet
from runtime.metal_serving import Model
assert 'dense_prefix_cache' not in garnet.backend_info()
s=garnet.decoder_session()
assert s.prefix_cache_bytes==0 and s.prefix_cache_entries==0 and s.prefix_restore_bytes==0 and s.prefix_capture_bytes==0
for name in ['capture_prefix','restore_prefix']:
 rejected=False
 try:getattr(s,name)(64)
 except Exception as error:
  assert 'not enabled' in str(error);rejected=True
 assert rejected
assert s.clear_prefix_cache()
rejected=False
try:Model('/Users/shawn/CantorAI/out/models/Qwen3-1.7B','','/Users/shawn/CantorAI/out/cache/Qwen3-1.7B',{'textPrefixCache':True},'Qwen3-1.7B')
except ValueError as error:
 assert 'requires the validated' in str(error);rejected=True
assert rejected
pathlib.Path('/Users/shawn/Documents/Codex/2026-10-15/i/work/worldsense-prefix-cache/default-off-guard.json').write_text(json.dumps({'feature_off_build':True,'unsupported_api_rejected':True,'logical_cache_empty':True,'invalid_profile_rejected_before_loading':True},indent=2))
print('Feature-OFF native API and profile rejection PASS')
