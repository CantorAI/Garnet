"""Adapt the proven archive layout to CORE source/counters, not residency artifacts."""
from pathlib import Path
import ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');folder=b/'work/worldsense-opt';text=(folder/'archive_ws69.py').read_text().replace('WS69','WS70').replace('WS-M5-69','WS-M5-70').replace('worldsense-packed-request-residency','worldsense-primary-core-isolation').replace('worldsense-m5-packed-request-residency','worldsense-m5-primary-core-isolation').replace('ws69','ws70')
start=text.index("report=(b/'outputs/");end=text.index("(d/'README.md')",start);text=text[:start]+"report=(w/'result.txt').read_text().replace('Canonical archival/publication pending.','Canonical evidence prepared for independent source/data/metric reconstruction before publication.')\n"+text[end:]
text=text.replace("['primary_projection_words.bin','packed_request_projection_words.bin','packed_request_projection_words.bin.leased.bin']","['primary_projection_words.bin','primary_core_projection_words.bin','primary_core_projection_words.bin.summary.json']").replace("'packed_lease_corpus_test'","'core_library_corpus_test'")
text=text.replace('calls=[];parents=[];routes=[];memory=[];cleanup_memory=[];leases=[];attempts=[];counts={}','calls=[];parents=[];routes=[];memory=[];core_rows=[];attempts=[];counts={}')
text=text.replace("['primary_storage_after','primary_encoded_delta','packed_request_weight_residency']","['primary_storage_before','primary_storage_after','primary_encoded_delta']")
start=text.index("    if'memory_after_packed_request_cleanup'");end=text.index('for name,rows in',start)
text=text[:start]+'''    available='primary_storage_before'in t
    derived={key:(t['primary_storage_after'][key]-t['primary_storage_before'][key]if available else None)for key in ['core_library_decode_encoded','core_library_prefill_encoded','core_library_head_encoded']}
    core_rows.append(dict(identity,counter_telemetry_available=available,**derived))
'''+text[end:]
text=text.replace("('post-cleanup-memory-samples.csv',cleanup_memory),('packed-request-lifecycle.csv',leases)","('core-dispatch-deltas.csv',core_rows)")
text=text.replace(" and len(cleanup_memory)==len(leases)==18"," and len(core_rows)==32 and sum(v['counter_telemetry_available']for v in core_rows)==22")
text=text.replace('post_cleanup_rows=len(cleanup_memory),packed_lifecycle_rows=len(leases)','core_delta_rows=len(core_rows),counter_available_rows=sum(v[\'counter_telemetry_available\']for v in core_rows)')
text=text.replace("memory='Original postdecode snapshot metric retained. Separate postcleanup snapshots show release, not a replacement metric. Not peaks/free/unique/physical-residency RAM. Legacy speed memory unavailable/null.'","memory='Original postdecode snapshots, not peaks/free/unique/physical-residency RAM. Legacy speed memory and core counters unavailable/null, never invented zero.'")
text=text.replace('Valid rejection: both model latency and4Bderiveddecode gates fail; wholeSkillwall within2%. No later stages/promotion.','Valid rejection: both model TTFT,4Btotal andwholeSkillwall fail unchanged2%gates, despite favorable observed deriveddecode. No later stages/promotion.')
text=text.replace('post_cleanup_rows=len(cleanup_memory)','core_delta_rows=len(core_rows)')
assert 'cleanup_memory'not in text and 'leases'not in text and 'packed_request_projection'not in text
ast.parse(text);p=folder/'archive_ws70.py';assert not p.exists();p.write_text(text);print('WS70 archiver prepared: original32calls/22memory rows and32core rows (legacy10 unavailable), source/native/provenance closure; no copy yet')
