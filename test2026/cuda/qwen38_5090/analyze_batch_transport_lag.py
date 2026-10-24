"""Observe native-to-bridge timestamp gaps; no GPU speed or clock-sync claim."""
from pathlib import Path
import json,sys,statistics
path=Path(sys.argv[1]);rows={}
for line in path.read_text(encoding="utf8").splitlines():
 try:
  wrapper=json.loads(line);event=json.loads(wrapper["line"])
 except (ValueError,KeyError):continue
 if event.get("event")!="token":continue
 rows.setdefault(event["id"],[]).append((wrapper["received"],event["native_monotonic_seconds"]))
result=[]
for key,events in rows.items():
 gaps=[received-native for received,native in events]
 result.append({"id":key,"observed_tokens":len(events),"first_gap_seconds":gaps[0],"last_gap_seconds":gaps[-1],"gap_growth_seconds":gaps[-1]-gaps[0],"median_gap_seconds":statistics.median(gaps),"native_timestamp_span_seconds":events[-1][1]-events[0][1],"bridge_receipt_span_seconds":events[-1][0]-events[0][0]})
print(json.dumps({"scope":"Timestamp gap observation only; cross-process clock offsets, tokenizer work, pipe transit and reader scheduling not causally separated. No speed claim.","requests":result},indent=2))
