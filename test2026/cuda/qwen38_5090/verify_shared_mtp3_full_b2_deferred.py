"""Replay full-model B2 IDs independently; verified failures never count as passes."""
from pathlib import Path, PureWindowsPath
import hashlib
import json
import zipfile

out = Path(__file__).parent
sha = lambda b: hashlib.sha256(b).hexdigest()
archive = out / "shared-mtp3-full-b2-deferred-attempt1-evidence.zip"
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None
    evidence = json.loads(z.read("evidence-manifest.json"))
    for n, h in evidence["files"].items(): assert sha(z.read(n)) == h, n
    for n, h in evidence["linked_archives"].items(): assert sha((out / n).read_bytes()) == h, n
    base = "results/shared-mtp3-full-b2-deferred-attempt1/"
    workflow = json.loads(z.read(base + "workflow.json"))
    report = json.loads(z.read(base + "native/report.json"))
    build = json.loads(z.read("results/native-full-attention-fp8-build-attempt1/manifest.json"))
    assert evidence["library_sha256"] == workflow["library_sha256"] == report["garnet_dll_sha256"] == build["library_sha256"]
    for n, h in build["source_sha256"].items(): assert sha(z.read("Garnet/" + n)) == h
    assert report["batch"] == 2 and report["capacity"] == 4096 and report["chunk"] == 16
    stage = json.loads(z.read("results/full-attention-policy-candidate-stages-attempt1-verified.json"))
    assert stage["status"] == "full_attention_policy_candidate_stages_verified"
    assert stage["all_171_arrays_match_retained_canonical_b1"] and stage["library_sha256"] == evidence["library_sha256"]
    with zipfile.ZipFile(out / "native-batch2-initial-failure-evidence.zip") as old:
        assert z.read("Garnet/test2026/cuda/qwen38_5090/probe_native_batch.py") == old.read("Garnet/test2026/cuda/qwen38_5090/probe_native_batch.py")
    baseline = json.loads(z.read("results/full-attention-policy-full-b2-attempt1-verified.json"))
    assert baseline['model_gate_passed'] and baseline['library_sha256'] == evidence['library_sha256']
    assert report['mtp_depth']==3 and report['draft_chunk']==4
    assert report['draft_builder_workspace_bytes']==128<<20
    assert workflow['draft_builder_workspace_bytes']==128<<20
    assert report['deferred_state_commit'] and workflow['deferred_state_commit']
    old_mtp=json.loads(z.read('results/shared-mtp3-full-b2-attempt2-verified.json'))
    assert old_mtp['model_gate_passed'] and old_mtp['strict_target_state_hash_gate_passed']
    assert old_mtp['library_sha256']==evidence['library_sha256']
    assert old_mtp['archive_sha256']==evidence['linked_archives']['shared-mtp3-full-b2-attempt2-evidence.zip']
    latest=json.loads(z.read('results/batch4-t4-http-pilot-attempt1-verified.json'))
    assert latest['status']=='batch4_t4_http_pilot_independently_verified' and latest['library_sha256']==evidence['library_sha256']
    assert latest['archive_sha256']==evidence['linked_archives']['batch4-t4-http-pilot-attempt1-evidence.zip']
    with zipfile.ZipFile(out/'shared-mtp3-full-b2-attempt2-evidence.zip') as old_zip:
        expected_source=old_zip.read('Garnet/test2026/cuda/qwen38_5090/probe_shared_mtp3_full_b2_workspace.py').decode().replace('\r\n','\n')
        for name in ('cuda_packed_decoder.py','cuda_packed_program.py','cuda_speculative.py','cuda_packed_replay.py','cuda_mtp.py','cuda_verify.py'):
            path='Garnet/xModel/qwen35/vl_0_8b/'+name
            assert z.read(path)==old_zip.read(path),path
    expected_source=expected_source.replace('retain_linear_inputs=True)','retain_linear_inputs=True,deferred_state_commit=True)')
    expected_source=expected_source.replace("'draft_builder_workspace_bytes':128<<20", "'deferred_state_commit':True,'draft_builder_workspace_bytes':128<<20")
    expected_source=expected_source.replace(' def state_hashes():', " def state_hashes():\n  assert decoder.state_transaction is None and not speculative.verifier.pending,'State hash requires closed transaction'")
    assert z.read('Garnet/test2026/cuda/qwen38_5090/probe_shared_mtp3_full_b2_deferred.py').decode().replace('\r\n','\n')==expected_source

    prior_failure=json.loads(z.read('results/shared-mtp3-full-b2-attempt1-verified.json'))
    assert prior_failure['archive_sha256']==evidence['linked_archives']['shared-mtp3-full-b2-attempt1-evidence.zip']
    assert prior_failure['status']=='shared_mtp3_full_b2_failed_gate_evidence_verified' and not prior_failure['model_gate_passed']
    assert not prior_failure['comparisons']
    for name,want in workflow['workspace_source_sha256'].items(): assert sha(z.read('Garnet/'+name))==want
    http_embedded='results/batch-http-pilot-attempt2-verified.json' in evidence['files']
    http=json.loads(z.read('results/batch-http-pilot-attempt2-verified.json')) if http_embedded else json.loads((out/'batch-http-pilot-attempt2-verified.json').read_text())
    assert http['archive_sha256']==evidence['linked_archives']['batch-http-pilot-attempt2-evidence.zip']
    assert http['status']=='batch_http_pilot_independently_verified' and http['library_sha256']==evidence['library_sha256']
    assert report['external_compressed_weights']
    pool = report['external_compressed_pool']
    assert pool['compressed_projection_pool_entries']==400 and pool['compressed_nvfp4_entries']==192 and pool['compressed_fp8_entries']==208
    physical = pool['compressed_projection_weight_bytes']+pool['compressed_projection_scale_bytes']
    launcher = z.read("test-shared-mtp3-full-b2-deferred.bat").decode()
    assert "set GARNET_FP8_FULL_ATTN_M32_CANONICAL=1" in launcher
    assert "set GARNET_FP8_GDN_M32_CANONICAL=1" in launcher
    keys = ["s3-text-c1-r0-greedy", "s3-image-c1-r0-greedy", "red", "green"]
    assert len(report["oracles"]) == 4 and {v["key"] for v in report["oracles"]} == set(keys)
    for oracle in report["oracles"]:
        for path, digest in oracle["oracle_sha256"].items():
            name = PureWindowsPath(path).relative_to(PureWindowsPath("D:/CantorAI/5090-study-20261023")).as_posix()
            assert sha(z.read(name)) == digest
    teacher = {}
    for key in keys:
        prefix = "results/native-mtp3-http/" + key
        choice = json.loads(z.read(prefix + "-response.json"))["choices"][0]
        request = json.loads(z.read(prefix + "-request.json"))
        teacher[key] = (choice, request)
    comparisons = []
    for case in report["cases"]:
        expected_keys = keys[:2] if case["group"] == 0 else keys[2:]
        assert case["keys"] == expected_keys and case["repeat"] in (0, 1)
        expected = [teacher[k][0]["response_token_ids"] for k in expected_keys]
        assert case["expected_ids"] == expected
        assert case['ordinary_ids']==expected
        assert case['full_target_state_exact']==(case['ordinary_state']==case['mtp_state'])
        for state in (case['ordinary_state'],case['mtp_state']):
            assert state['state_words_per_row']>0 and len(state['state_row_sha256'])==2
            assert all(len(h)==64 and all(c in '0123456789abcdef' for c in h) for h in state['state_row_sha256']+[state['last_hidden_sha256']])
            assert len(state['slots'])==2
        assert case['mtp_memory']['target_prefill_chunk']==16 and case['mtp_memory']['draft_chunk']==4
        assert case['decoder_memory']['packed_profiles']['deferred_state_commit']
        assert case['mtp_memory']['target_verifier']['deferred_state_commit']
        assert case['mtp_memory']['target_verifier']['canonical_state_backup_bytes']==0

        metrics=case['speculative_metrics']
        assert metrics['target_forward_calls']==metrics['target_verification_calls']+metrics['target_prefix_recomputes']
        assert metrics['target_verification_calls']==len(metrics['iterations'])
        assert 0<=metrics['accepted_draft_tokens']<=metrics['proposed_draft_tokens']
        assert case["prompt_lengths"] == [len(teacher[k][0]["prompt_token_ids"]) for k in expected_keys]
        assert case["limits"] == [teacher[k][1]["max_tokens"] for k in expected_keys]
        assert len(case["generated_ids"]) == 2
        bodies = [g for g in case["decoder_memory"]["graphs"] if g["native_nvfp4_projection_weights"] == 192 and g["native_fp8_projection_weights"] == 208]
        for graph in bodies:
            assert graph['external_bf16_weight_inputs']==0 and graph['external_compressed_weight_inputs']==400
            assert graph['external_compressed_weight_bytes']==physical
        assert len(bodies) == 1 and bodies[0]["checkpoint_weight_precision"] == "modelopt_mixed"
        assert bodies[0]["graph_capture_failures"] == 0 and bodies[0]["graph_captures"] >= 1 and bodies[0]["graph_replays"] >= 1
        exact = case["generated_ids"] == expected
        assert case["exact_b1_ids_match"] == exact
        for key, actual, reference in zip(expected_keys, case["generated_ids"], expected):
            assert 0 < len(actual) <= teacher[key][1]["max_tokens"]
            first = next((i for i, (a, b) in enumerate(zip(actual, reference)) if a != b), None)
            if first is None and len(actual) != len(reference): first = min(len(actual), len(reference))
            comparisons.append({"key": key, "repeat": case["repeat"], "exact": actual == reference,
                "generated_count": len(actual), "expected_count": len(reference), "first_differing_index": first})
    for directory in (base + "ledger/",):
        events = [json.loads(l) for l in z.read(directory + "commands.jsonl").decode().splitlines()]
        assert events[-1]["event"] == "finished" and events[-1]["exit_code"] == workflow["exit_code"]
    monitor = json.loads(z.read(base + "resources/resource-monitor.json"))
    samples = [json.loads(l) for l in z.read(base + "resources/resource-vram.jsonl").decode().splitlines()]
    assert monitor["child_exit_code"] == workflow["exit_code"]
    assert monitor["sampled_memory"]["samples"] == len(samples)
    assert monitor["sampled_memory"]["sampled_peak_adapter_used_bytes"] == max(v["used_bytes"] for v in samples)
    passed = workflow["exit_code"] == 0
    if passed:
        assert report["status"] == "backend_smoke_completed" and len(report["cases"]) == 4
        assert {(v["group"], v["repeat"]) for v in report["cases"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
        assert len(comparisons) == 8 and all(v["exact"] for v in comparisons)
        assert all(case["full_target_state_exact"] for case in report["cases"])
    else: assert report["status"] == "failed"
    result = {"status": "shared_mtp3_full_b2_deferred_id_state_hash_gate_verified" if passed else "shared_mtp3_full_b2_deferred_failed_gate_evidence_verified",
        "model_gate_passed": passed, "archive_sha256": sha(archive.read_bytes()), "files_verified": len(evidence["files"]),
        "library_sha256": evidence["library_sha256"], "comparisons": comparisons,
        "loaded_module_path_observed": False, "build_source_hashes_verified": True,
        "strict_target_state_hash_gate_passed": passed, "raw_target_state_arrays_retained": False, "http_dependency_proof_embedded": http_embedded,
        "performance_admitted": False,
        "scope": "B2 MTP3 retained-ID plus producer-observed full target-state hash witness; raw target-state arrays are not retained. Source coverage and hash equality replay, not independent raw-state arithmetic replay. Failure is preserved as failure. No HTTP, MTP, broad quality, decode speed, true concurrency throughput or2x claim."}
(out / "shared-mtp3-full-b2-deferred-attempt1-verified.json").write_text(json.dumps(result, indent=2), encoding="utf8")
print(json.dumps(result, indent=2))
