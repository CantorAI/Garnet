"""Prepare archive-only exact-source/data/numerical verification for group2 result."""
from pathlib import Path
import ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');f=b/'work/worldsense-opt'
def adapt(t):
 return t.replace('worldsense-primary-decode-groups8','worldsense-primary-decode-groups2').replace('worldsense-m5-primary-decode-groups8','worldsense-m5-primary-decode-groups2').replace('PRIMARY_DECODE_GROUPS8','PRIMARY_DECODE_GROUPS2').replace('primary_decode_groups8','primary_decode_groups2').replace('PrimaryDecodeGroups8','PrimaryDecodeGroups2').replace('decodeGroups8','decodeGroups2').replace('headGroups8','headGroups2').replace('groups8','groups2').replace('ws72','ws73').replace('WS72','WS73').replace('WS-M5-72','WS-M5-73').replace('ulong column=ulong(group)*32+simd*4;','ulong column=ulong(group)*8+simd*4;').replace('[260,272,288,1024]','[260,264,288,1024]').replace('[260,272]','[260,264]').replace('[100352,100368]','[100360,100356]').replace('804504','803580')
t=adapt((f/'archive_ws72.py').read_text()).replace('Valid rejection: bothmodels TTFT/total/deriveddecode andwholeSkill fail unchanged2%gates; no laterstages/promotion.','Valid rejection:4Bderiveddecode andalltextmetrics fail unchanged2%gates;4Blatency/wholeSkillfavorable butnorescue. No laterstages/promotion.')
ast.parse(t);p=f/'archive_ws73.py';assert not p.exists();p.write_text(t)
t=adapt((f/'verify_archive_ws72.py').read_text()).replace("len(load(d/'flag-dependency-proof.json'))==24","len(load(d/'flag-dependency-proof.json'))==25").replace('==23617','==23692').replace('offline_malformed_copies_rejected=23617','offline_malformed_copies_rejected=23692')
line=next(line for line in t.splitlines()if "assert load(d/'pre-native-generic-fixture-review/correctness-source-manifest.json')"in line);t=t.replace(line,"assert hashlib.sha256((d/'primary_decode_groups2_test.cpp').read_bytes()).hexdigest()==load(d/'correctness-source-manifest.json')['test2026/metal/primary_decode_groups2_test.cpp']")
ast.parse(t);p=f/'verify_archive_ws73.py';assert not p.exists();p.write_text(t)
print('WS73 archive andcanonical verifier ready: originalfixed archive-only72structure, no unusedlocalbuild binding, no phantom73fixture repair; sourceinverse/new60words/25configs/23692negatives/32pipelinecalls/groups2CSV. No copy/inference yet.')
