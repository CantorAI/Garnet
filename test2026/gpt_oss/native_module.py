"""XLang3 native import/bridge contract; runs without a plugin registry."""
import garnet
import garnet_gpt_oss
import json
manifest = json.loads(garnet_gpt_oss.manifest_json())
assert manifest['id'] == 'gpt_oss' and manifest['abi'] == 1
assert garnet.bind_operator_module(garnet_gpt_oss)
assert garnet.bind_operator_module(garnet_gpt_oss)
failed = False
try:
    garnet.bind_operator_module(json)
except Exception:
    failed = True
assert failed, 'ordinary modules must not be accepted as operator modules'
print('native-module-import-and-bridge-passed', flush=True)
