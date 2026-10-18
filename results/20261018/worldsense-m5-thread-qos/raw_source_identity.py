"""Read-only shader-source/flag comparison; no GPU execution or speed claim."""
from pathlib import Path
import re,json,hashlib,datetime
w=Path(__file__).resolve().parent;r=Path('/Users/shawn/CantorAI/out');old=r/'research/worldsense-prefix-cache/source';new=r/'research/worldsense-native-text-worker/source';archive=Path('/Users/shawn/CantorAI/Garnet/results/20261017/worldsense-m5-native-text-worker');flags={}
for name,p in [('WS10',r/'build/garnet-metal-worldsense-prefix-cache/CMakeCache.txt'),('RAW58',r/'build/garnet-metal-worldsense-native-text-worker-off/CMakeCache.txt')]:
 values={}
 for line in p.read_text().splitlines():
  if line.startswith('GARNET_')and':BOOL='in line:k,v=line.split(':BOOL=');values[k]=v
 flags[name]=values
assert all(flags['WS10'][k]==flags['RAW58'][k]for k in flags['WS10']if k in flags['RAW58'])
raw={name:re.findall(r'R"METAL\((.*?)\)METAL"',(root/'src/backends/metal/metal_context.mm').read_text(),re.S)for name,root in [('WS10',old),('RAW58',new)]};assert len(raw['WS10'])==len(raw['RAW58'])==3 and raw['WS10']==raw['RAW58']
body=(old/'src/backends/metal/body_shape_shader.h').read_text();assert body==(new/'src/backends/metal/body_shape_shader.h').read_text();body=body.split('R"BODYSHAPE(',1)[1].split(')BODYSHAPE"',1)[0]
# For these measured flags, constructor additions are ATTENTION_PREFIX and
# BODY_SHAPE only. All optional string transforms/additions are disabled.
for key in ['GARNET_METAL_FUSED_VISION_CONTEXT','GARNET_METAL_PARALLEL_VISION_CONTEXT','GARNET_METAL_JOINED_PREFILL','GARNET_METAL_NATIVE_BFLOAT_BODY','GARNET_METAL_CONSTANT_BF16_HEAD','GARNET_METAL_SINGLE_CODER_HEAD','GARNET_METAL_INTERLEAVED_BF16_BODY','GARNET_METAL_COALESCED_BF16_BODY']:
 assert flags['WS10'].get(key,'OFF')=='OFF'
assert flags['WS10']['GARNET_METAL_ATTENTION_PREFIX']=='ON'and flags['WS10']['GARNET_METAL_BODY_SHAPE']=='ON'
assembled=('#define GARNET_ATTENTION_PREFIX 1\n'+raw['WS10'][0]+body).encode();actual=(archive/'actual-off-main-program.metal').read_bytes();assert assembled==actual
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),WS10_source_files=1706,RAW58_source_files=1723,all_common_original_bool_flags_equal=True,all_three_original_raw_literal_blocks_exact=True,body_shape_header_exact=True,assembled_WS10_MAIN_equals_actual_RAW58_MAIN=True,main_bytes=len(actual),main_sha256=hashlib.sha256(actual).hexdigest(),references=dict(old_source=str(old),new_source=str(new),actual_constructor_dump=str(archive/'actual-off-main-program.metal')),scope='Read-only source composition and actual new constructor dump identity. Old source composition follows its pinned build flags. No new GPU job, binary-ISA identity, runtime resource identity, scheduling equivalence or causal/performance claim.')
(w/'raw-source-identity.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
