"""Declare supplemental original proof routes before any extra builds."""
from pathlib import Path
import datetime,hashlib,json,re
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-paired-gpu-profile';r=Path('/Users/shawn/CantorAI')
assert json.loads((w/'native-independent-audit.json').read_text())['native_CTest_count']==39
assert not (w/'compatibility-proof-protocol.json').exists()
text='''WS76 full-proof addendum, before supplemental builds or any learned execution

The six initial paired-profile native builds passed39CTest cases, original186
complete projection streams and current20additional records, actual program
strings and counter transaction checks. Independent audit passed after the
retained CTest parser-only post-terminal repair. No model/capture has run.

Full coverage must retain the ORIGINAL WS74 proof routes, including RAW,
PRIMARY, ON, FAULT, COMPAT56 and COMPAT56FAULT and old raw-request lifecycle.
The initially declared six builds cover legacy quiet/trace/counterfault and
current ONquiet/ONtrace/ON+primary+counterfault. They are insufficient alone
to execute every original compatibility proof. Add four current-source
correctness-only roles: current-raw, current-primary, current-compat and
current-compatfault. Exactly reuse the immutable original74 cache flags for
off,primary,compat56,compat56-fault respectively, then enable the common
counter diagnostic in allfour. Dedicated counter-test faults ON only for
compatfault; original primary/request fault flags remain exactly as in74.
Source, helper/test, models, math, oracles, limits and all gates unchanged.
Neither new role is used for actual diagnostic timing/capture or promotion.
This supplements the original six-role plan, never narrows original coverage.

Before learned execution, run all nine original current native tests plus
the common counter test in each additional role; check all186complete
projection streams,20additional records/432804words and actual MAIN/prefill/
emptyCORE programs against EACH matching original74role. Common native
observer is byte-exact to initial76source. Preserve original faults and
4GiBraw-request controls. Freeze full inputs/outer auditor before dispatch;
unconditionally restore normalWS10 after terminal. Then adapt full learned
controllers only for source/role path and explicit trace admission, retaining
all original quantitative references, allstates/tokens/vision/lifecycles and
five packed readers/201matrices plus raw/direct coverage. Legacy also needs
trace/quiet/fault API/fullstate/error coverage, not invented packed counters.

Original paired16call/mirrored4worker capture design and analysis unchanged.
All original optimization gates remain unchanged. No speed acceptance or
causal claim is possible from this observer diagnostic. Full goalACTIVE.
'''
(w/'full-proof-addendum-before-supplemental-build.txt').write_text(text)
(b/'outputs/Garnet-4B-WS76-Full-Proof-Addendum.txt').write_text(text)
mapping={'current-raw':'off','current-primary':'primary','current-compat':'compat56','current-compatfault':'compat56-fault'}
roles={}
for role,old in mapping.items():
    p=r/('out/build/garnet-metal-worldsense-minimal-selective-defaults-'+old)/'CMakeCache.txt'
    flags=dict(re.findall(r'^(GARNET_METAL(?:_[A-Z0-9_]+)?):BOOL=(ON|OFF)$',p.read_text(),re.M))
    flags.update(GARNET_METAL_GPU_PASS_COUNTER_PROFILE='ON',GARNET_METAL_GPU_PASS_COUNTER_TEST_FAULTS='ON' if role=='current-compatfault' else 'OFF')
    roles[role]=dict(original_role=old,flags=flags,original_cache=str(p),original_cache_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
record=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),design_sha256=hashlib.sha256(text.encode()).hexdigest(),roles=roles,no_source_change=True,no_learned_model_or_capture_yet=True,original_full_coverage_supplement=True,promotion=False)
(w/'compatibility-proof-protocol.json').write_text(json.dumps(record,indent=2)+'\n')
print('Supplemental four current original-proof routes declared; no source change/build/model run')
