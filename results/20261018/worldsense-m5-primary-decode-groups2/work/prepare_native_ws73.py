"""Prepare original complete native proof plus distinct2group tail/dispatch checks."""
from pathlib import Path
import ast,shutil
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt';old=b/'work/worldsense-primary-decode-groups8';w=b/'work/worldsense-primary-decode-groups2'
def adapt(t):
 return t.replace('worldsense-primary-decode-groups8','worldsense-primary-decode-groups2').replace('PRIMARY_DECODE_GROUPS8','PRIMARY_DECODE_GROUPS2').replace('primary_decode_groups8','primary_decode_groups2').replace('PrimaryDecodeGroups8','PrimaryDecodeGroups2').replace('decodeGroups8','decodeGroups2').replace('headGroups8','headGroups2').replace('groups8','groups2').replace('ws72','ws73').replace('WS72','WS73').replace('ulong column=ulong(group)*32+simd*4;','ulong column=ulong(group)*8+simd*4;').replace('[260,272,288,1024]','[260,264,288,1024]').replace('[260,272]','[260,264]').replace('[100352,100368]','[100360,100356]').replace('804504','803580')
shutil.copy2(old/'primary_core_isolation_test.cpp',w/'primary_core_isolation_test.cpp')
t=adapt((old/'native_controller.py').read_text());t=t.replace("==(w/'rebuild-controller.log').read_text().splitlines()",'').replace('pre-native-fixture-amendment.txt','reused-native-fixture-note.txt').replace("'repair_fixture_groups2_ws73.py',", "'prepare_groups2_implementation_ws73.py',")
ast.parse(t);(w/'native_controller.py').write_text(t)
v=adapt((f/'verify_native_ws72.py').read_text());ast.parse(v);p=f/'verify_native_ws73.py';assert not p.exists();p.write_text(v)
print('WS73 native60/original186+20 andnew60complete803580words prepared, unique raw512, no phantomrebuild or73repairclaim. No native/model yet.')
