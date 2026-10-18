"""Prepare same original learned proof and future gates for declared group2 profile."""
from pathlib import Path
import ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt'
def adapt(t):
 return t.replace('worldsense-primary-decode-groups8','worldsense-primary-decode-groups2').replace('PRIMARY_DECODE_GROUPS8','PRIMARY_DECODE_GROUPS2').replace('primary_decode_groups8','primary_decode_groups2').replace('PrimaryDecodeGroups8','PrimaryDecodeGroups2').replace('decodeGroups8','decodeGroups2').replace('headGroups8','headGroups2').replace('groups8','groups2').replace('ws72','ws73').replace('WS72','WS73').replace('pre-native-fixture-amendment.txt','reused-native-fixture-note.txt')
t=adapt((f/'prepare_full_ws72.py').read_text());ast.parse(t);p=f/'prepare_full_ws73.py';assert not p.exists();p.write_text(t)
t=adapt((f/'prepare_pipeline_helpers_ws72.py').read_text())
needle="t=t.replace(\"'core_library_head_encoded'}\",\"'core_library_head_encoded','decode_groups2_encoded','head_groups2_encoded'}\")";assert t.count(needle)==1
t=t.replace(needle,needle+"\n+t=t.replace(\"['request_scope_weight_residency','packed_request_weight_residency'\",\"['primary_decode_groups8','request_scope_weight_residency','packed_request_weight_residency'\")")
# The inserted line above is Python source, not patch syntax.
t=t.replace('\n+t=t.replace','\nt=t.replace')
t=t.replace('WS73 packed decode8SIMD/256threads/32outputs, exact2origin substitutions andnoTail32','WS73 packed decode2SIMD/64threads/8outputs, exact2origin substitutions andnoTail8')
t=t.replace("new9_groups2_combinations_rejected=True","new10_groups2_combinations_rejected=True")
t=t.replace('len(results)==24','len(results)==25').replace('original7+8core+9groups2','original7+8core+10groups2')
t=t.replace("'Packed groups2 excludes CORE and raw request residency'","'Packed groups2 excludes CORE, raw request residency and GROUPS8'")
needle="cases += [('groups2-primary'";assert t.count(needle)==1
# Add the explicitly incompatible old8flag before the nine existing new cases.
t=t.replace(needle,"cases += [('groups2-old8',groups+['-DGARNET_METAL_PRIMARY_DECODE_GROUPS8=ON'],'Packed groups2 excludes CORE, raw request residency and GROUPS8')]\n"+needle)
t=t.replace("'primary_decode_groups2','request_scope_weight_residency'","'primary_decode_groups2','primary_decode_groups8','request_scope_weight_residency'").replace("'primary_decode_groups2','packed_request_weight_residency'","'primary_decode_groups2','primary_decode_groups8','packed_request_weight_residency'").replace('and24CMake exclusions','and25CMake exclusions')
ast.parse(t);p=f/'prepare_pipeline_helpers_ws73.py';assert not p.exists();p.write_text(t)
print('WS73 original fulllearned/future stage scaffolds prepared. Additional incompatibleGROUPS8configuration/capability will be rejected. No model or dependent stage run.')
