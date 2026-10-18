from pathlib import Path
import datetime, json

b = Path('/Users/shawn/Documents/Codex/2026-10-15/i')
w = b / 'work/worldsense-cooperative-prefill'
load = lambda f: json.loads(f.read_text())
a = load(w / 'primary-speed/independent-stage-audit.json')
assert a['decision'] == 'valid-reject' and a['exit_code'] == 0
screen = load(w / 'production-speed/independent-screen-audit.json')
report = '''WS-M5-67: cooperative interleaved prefill — VALID REJECT

The optimization goal remains active. Keep validated WS10. This experimental profile passed native and learned numerical proofs, actual WorldSense functional checks, resource gates and the first deployed-WS10 speed screen. It failed the subsequent mandatory matched-PRIMARY screen. No promotion, unchanged retry, RAW20, mirrored48 or canonical implementation integration is authorized by these results.

Method: default-OFF cooperative sole-MLP prefill addressing and padded right-hand shared tiles, with the WS66 dual-projection decode profile retained. Four original prefill/tile-reader kernels use a shared integer coordinate mapping; right-hand row strides increase from N to N+4. The 84 sole text MLP weights, numerical order, complete outputs, vision path, original model configuration, CUDA and XLang3 engine remain preserved. This is a whole-profile comparison; no isolated cooperative-kernel gain or cause is established.

Evidence: seven builds passed 63 native tests, complete 186 original projection, 28 MLP and 36 paired records. Original full learned state/token/weight oracles pass, including the 56-matrix paired GPU reader and 476 real resident/profile route checks. All 25 expected invalid configurations are rejected. Four real WorldSense/Galaxy functional calls preserve complete original outputs. Native C++/Metal/XLang3 inference and image handling; CPython is orchestration/audit only. No camera.

Separate resource cohort, 8 actual model calls: post-decode sampled Metal allocation 21,903,376,384 -> 16,870,899,712 bytes (-22.975803%); process footprint 18,074,598,904 -> 16,560,830,504 bytes (-8.375115%). Original 10%/5% resource gates pass. These are whole-profile snapshots versus RAW, not peaks or isolated cooperative-memory savings.

First deployed-WS10 screen, 20 actual model calls: 4B warm median TTFT 9.417305 -> 5.063230 seconds and whole-Skill wall 21.929449 -> 17.205226 seconds. The original 2% preservation screen passes. These provisional results do not override the following rejection.

Mandatory matched-PRIMARY screen (session 58249, exit 0): 20 actual model calls in two fresh sequential workers, five requests per worker; first retained separately, four warm samples/model/worker. Original exact outputs, native routes, input identities and thresholds are preserved.

Metric                                      PRIMARY control      Candidate
'''
for model in ['Qwen3.5-4B', 'Qwen3-1.7B']:
    for key, label in [('ttft', 'TTFT seconds'), ('total', 'total seconds'), ('decode_tps', 'derived decode tokens/s')]:
        report += f"{model + ' ' + label:<44}{a['metrics']['control-1'][model][key]:>14.6f}{a['metrics']['candidate-1'][model][key]:>15.6f}\n"
report += f"{'Complete Skill wall seconds':<44}{a['whole_skill_wall']['control-1']:>14.6f}{a['whole_skill_wall']['candidate-1']:>15.6f}\n"
report += '''
4B TTFT regressed 34.768356%, total 49.370746%, derived decode 38.045227%. Text TTFT regressed 75.678442%, total 52.187665%, derived decode 13.192391%. Whole-Skill latency regressed 49.923671%. The required >=5% 4B TTFT gain and <=2% preservation limits fail. Sampled Metal allocation is equal; process footprint rises from 16,081,630,936 to 16,564,467,848 bytes (+3.002413%), also exceeding 2%. No selective favorable metric or earlier cohort rescues this candidate.

Independent post-terminal audit recomputed all 20 calls, exact original outputs, native admission, medians, memory and original gates, and rechecked source/runtime/library/drivers/19 learned assets/1,881 build and host dependencies. The earlier 32 actual functional/resource/speed calls are independently audited separately. No new inference, exclusions or retimed samples in either audit. All first/warm raw JSON and controller logs remain in the experiment work directory; canonical research archive/publication is pending.

Limits: derived decode=(generated-1)/(total-TTFT) includes host, stopping, post-loop, possible EOS and memory snapshot overhead; not pure GPU. Fixed-frame Skill calls do not cover live video decoding, scheduling, event persistence, shared states, retrieval/ranking, navigation or independent-camera fusion. First requests are not proven hardware-cold. Sequential whole-profile cohorts do not establish isolated kernel, hardware, VM, thermal or clock causes. Abstract layout proofs are not M5 bank/ISA/occupancy measurements.

Service restoration: validated WS10 launchd Cantor PID 89373, four video-file sources online, camera unmounted, valid benchmark POST returns 404 in normal mode. No benchmark remains running after the final audit. Next: archive and publish this valid rejection, then inspect retained phase evidence before a distinct prospective optimization. Keep every original confirmation and canonical-integration gate.
'''
(w / 'result.txt').write_text(report)
(b / 'outputs/Garnet-4B-WS67-Result.txt').write_text(report)
(b / 'outputs/4B-Latest-Result.txt').write_text(report)
(w / 'decision.json').write_text(json.dumps(dict(a, experiment='WS-M5-67', canonical_archive_pending=True), indent=2)+'\n')
(w / 'NEXT_IMPLEMENTATION.txt').write_text('WS67 CLOSED valid rejection at mandatory matchedPRIMARY20. Native/full/functional/resource/actualWS10 screen passed; matchedPRIMARY performance and footprint fail. No unchanged retry or later stages. Normal WS10 restored PID89373/four files/no camera/valid benchmark404. Next archive/audit/publish, then read-only retained-phase and prior-method audit before a distinct prospective optimization. Goal remains active.\n')
p = b / 'work/worldsense-next.json'
state = load(p)
state.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    state='CLOSED_VALID_REJECT', live_job=None,
    matched_primary_session=58249, matched_primary_exit_code=0,
    matched_primary_passed=False, independent_primary_audit_pass=True,
    independent_primary_audit_session=74303,
    promotion=False, publication_pending=True, goal_complete=False,
    normal_after_primary=dict(pid=89373, four_files_online=True, no_camera=True, valid_benchmark_POST404=True),
    next_actions=['Archive complete WS67 first/warm/native/full evidence, independently audit and publish paper logs.',
                  'Then read-only retained-phase and prior-method audit before DISTINCT prospective optimization; no unchanged retry.'])
p.write_text(json.dumps(state, indent=2)+'\n')
print('WS67 valid rejection, user-facing report and continuation checkpoint saved.')
