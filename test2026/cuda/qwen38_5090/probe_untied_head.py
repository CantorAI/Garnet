"""Run with the built XLang3/Garnet runtime after make_head_fixtures.py."""
import sys, os, json
sys.path.insert(0,os.environ['GARNET_RUNTIME_ROOT'])
import garnet
root, output = sys.argv[1:3]
io = garnet.cuda_tensor_io()
checks = []
try:
    for kind, multiplier in [('untied',2),('tied',1),('untied',2)]:
        io.open(os.path.join(root,kind))
        embed = garnet.tensor_to_cpu(io.embedding([3])).tolist()
        assert embed == [3.]*64
        hidden = io.from_host([1.]+[0.]*63,[1,64])
        logits = garnet.tensor_to_cpu(io.logits(hidden)).tolist()
        assert logits == [float(i*multiplier) for i in range(32)], logits
        info = json.loads(io.memory_json())
        assert info['embedding_head_tied'] == (kind == 'tied')
        assert info['untied_head_bytes'] == (0 if kind == 'tied' else 4096)
        checks.append({'case':kind,'exact_logits':True,'embedding_unchanged':True})
        io.close()
    for kind in ('packed','missing'):
        rejected = False
        try: io.open(os.path.join(root,kind))
        except Exception: rejected = True
        assert rejected, 'Packed or missing untied head must not silently reuse embedding'
        assert json.loads(io.memory_json())['embedding_table_bytes'] == 0
        checks.append({'case':kind+'_head_rejected_before_embedding_allocation'})
finally: io.close()
with open(output,'w') as f: f.write(json.dumps({'checks':checks,'scope':'tiny synthetic native correctness; no model or performance claim'},indent=2))
print('UNTIED_HEAD_NATIVE_CHECKS_PASS')
