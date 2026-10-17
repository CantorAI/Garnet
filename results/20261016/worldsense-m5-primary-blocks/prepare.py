from pathlib import Path
import json,collections,hashlib,shutil,datetime,subprocess
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI');base=r/'out/research/worldsense-primary-blocks/cpu-source';assert not base.exists();base.mkdir(parents=True)
old=w.parent/'worldsense-physical-memory';analysis=json.loads((old/'memory-analysis.json').read_text());o=next(x for x in analysis['observations']if x['worker']=='control-1'and x['model']=='Qwen3-1.7B'and not x['warmup']);capture=old/'observations'/o['worker']/o['capture_id']/'native-inventory.json';inv=json.loads(capture.read_text());groups=collections.defaultdict(list);buffers={b['id']:b for b in inv['buffers']}
for ref in inv['references']:
 if ref['representation']=='bf16':groups[ref['buffer_id']].append(ref)
matrices=[]
for bid,refs in groups.items():
 models={next(name for name in ['Qwen3-1.7B','Qwen3.5-4B']if x['tag'].startswith(name+'.'))for x in refs};names={x['name'].removesuffix('|BF16')for x in refs};shapes={tuple(x['shape'])for x in refs};assert len(models)==len(names)==len(shapes)==1,(bid,models,names,shapes)
 model=next(iter(models));shape=next(iter(shapes));assert len(shape)==2 and shape[0]*shape[1]*2==buffers[bid]['length_bytes']
 matrices.append({'model':model,'checkpoint':str(r/'out/models'/model),'tensor':next(iter(names)),'rows':shape[0],'inner':shape[1],'original_buffer_length_bytes':buffers[bid]['length_bytes'],'original_allocated_bytes':buffers[bid]['allocated_bytes'],'inventory_buffer_id':bid})
assert len(matrices)==446 and sum(x['original_buffer_length_bytes']for x in matrices)==11850481664
matrices.sort(key=lambda x:(x['model'],x['tensor']));(w/'matrix-selection.json').write_text(json.dumps({'selection':'All446actual globally deduplicated native BF16 weight buffers in frozen WS35 post-both-models control capture','inventory_path':str(capture),'inventory_sha256':hashlib.sha256(capture.read_bytes()).hexdigest(),'matrices':matrices},indent=2))
for src,dest in [(r/'Garnet/src/model/safetensors_index.h',base/'safetensors_index.h'),(r/'Garnet/src/model/safetensors_index.cpp',base/'safetensors_index.cpp'),(r/'Garnet/third_party/nlohmann/json.hpp',base/'nlohmann/json.hpp')]:
 dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dest)
(w/'protocol.json').write_text(json.dumps({'experiment':'WS-M5-36 feasibility','declared_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'purpose':'Evaluate distinct sole-primary lossless compact/raw BF16 block storage against measured graphics footprint, before GPU integration. No inference/performance claim.','prior_evidence':'Coalesced-body-native/block achieved exact values and some small-model decode gains but retained raw master, interleaved and compact copies. Hybrid byte estimates existed; no sole-primary execution identified. WS-M5-03 shared-half removed a master but failed full-state equivalence and performance; WS-M5-04/05 coalesced prefill failed WorldSense performance. Retain those decisions.','format':'Four output rows x128K,512BF16 words. If exponent range<=15:512sign/mantissa bytes+256exponent nibbles; otherwise1024raw interleaved bytes. One variable payload and one8byte metadata perblock, with256-byte-unit offset, compact flag and exponent base. No retained raw/interleaved second full copy. Unsupported dimensions or non-saving matrices keep original representation.','selection':'All446actualWS35globalBF16buffers,249VLM+197text including vocabulary heads;11,850,481,664rawbytes. VisionFP32/unrelatedslots excluded. No first-layer extrapolation.','correctness':'NativeCPU encode/reconstruct every original selected BF16 word; exhaustive65280finite words plus signedzero/subnormal/mixed-exponent/15-versus16/tail/metadata bounds tests; reject nonfinite data. Freeze all matrices, model files, source, compiler and per-tensor hashes. No arithmetic is performed on reconstructed values, no quantization.','accounting':'Report original bytes, compact/raw blocks, variable payload+metadata, unsupported/non-saving fallback, page-rounded estimates, encoding hashes and source identities. These are codec/logical-byte feasibility counts, not actual Metal allocation/physical RAM/latency or a WorldSense speed result.','next_gate':'Only if exactness/budgets feasible, design separate default-OFF nativeMetal support for decode+prefill preserving original FP32 operation order, generic fallbacks and CUDA, then native/fullstate correctness and freshWorldSense resource/speed/two-full-pair gates before promotion. No favorable unchanged retry.'},indent=2))
# Rehash original model assets before scan; never stage weights.
expected=json.loads((old/'model-assets.json').read_text());observed={}
for model,files in expected.items():
 observed[model]={}
 for rel,want in files.items():
  p=r/'out/models'/model/rel;h=hashlib.sha256()
  with p.open('rb')as stream:
   for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
  actual={'bytes':p.stat().st_size,'sha256':h.hexdigest()};assert actual==want
  observed[model][rel]=actual
(w/'model-assets.json').write_text(json.dumps(observed,indent=2))
print('Frozen actual446matrix selection and verified original model assets',flush=True)
