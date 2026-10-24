"""Research B2 MTP3 shared-weight streaming worker; GPU/HTTP gates precede timings."""
import base64
import hashlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.environ['GARNET_RUNTIME_ROOT'])
sys.path.insert(0, os.getcwd())
import garnet
from xModel.qwen38.vl_27b.model import validate_cuda_batch_study_profile, create_cuda_vision, prepare_cuda_image
from xModel.qwen35.vl_0_8b.cuda_packed_decoder import CUDAPackedDecoder
from test2026.cuda.qwen38_5090.native_prompt import image_positions
from test2026.cuda.qwen38_5090.native_worker_io import requests, emit

root = Path(sys.argv[1])
checkpoint = str(root / 'models/Qwen3.8-27B-NVFP4')
config = json.loads(Path(checkpoint, 'config.json').read_text())
assert os.environ.get('GARNET_FP8_FULL_ATTN_M32_CANONICAL') == '1'
assert os.environ.get('GARNET_FP8_GDN_M32_CANONICAL') == '1'
proof = json.loads((root / 'results/shared-full-b2-attempt1-verified.json').read_text())
library_sha = hashlib.sha256((root / 'build/bin/garnet.dll').read_bytes()).hexdigest()
assert proof['status'] == 'shared_full_b2_id_gate_verified' and proof['model_gate_passed']
assert proof['library_sha256'] == library_sha
validate_cuda_batch_study_profile(config, 4096, 2)
tokenizer = garnet.tokenizer()
tokenizer.open(checkpoint)
stops = [tokenizer.token_id('<|im_end|>'), tokenizer.token_id('<|endoftext|>')]
suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
decoder = None
vision = None
speculative = None
try:
    decoder = CUDAPackedDecoder(checkpoint, config['text_config'], 2, 4096, 16, 16,
        flat_concat=True, external_compressed_weights=True, retain_linear_inputs=True)
    decoder.io.enable_nvfp4_head(True)
    pool = json.loads(decoder.io.memory_json())
    assert pool['compressed_projection_pool_entries'] == 400
    assert pool['compressed_nvfp4_entries'] == 192 and pool['compressed_fp8_entries'] == 208
    physical = pool['compressed_projection_weight_bytes'] + pool['compressed_projection_scale_bytes']
    for program in decoder.programs.values():
        stats = json.loads(program.graph.stats_json())
        assert stats['external_compressed_weight_inputs'] == 400 and stats['external_bf16_weight_inputs'] == 0
        assert stats['external_compressed_weight_bytes'] == physical
    decoder.enable_cuda_graph(True)
    vision = create_cuda_vision(checkpoint, config, 16, 16)
    vision.graph.enable_cuda_graph(True)
    io = decoder.io
    mtp_proof = json.loads((root / 'results/shared-mtp3-full-b2-attempt2-verified.json').read_text())
    assert mtp_proof['model_gate_passed'] and mtp_proof['strict_target_state_hash_gate_passed']
    assert mtp_proof['status'] == 'shared_mtp3_full_b2_id_state_hash_gate_verified'
    assert mtp_proof['library_sha256'] == library_sha
    from xModel.qwen35.vl_0_8b.cuda_speculative import CUDASpeculativeSession
    speculative = CUDASpeculativeSession(decoder, checkpoint, stops, 3, head_candidates=0,
        prefix_replay=True, draft_chunk_size=4, draft_builder_workspace_bytes=128<<20)
    emit({'event': 'ready', 'protocol': 'study_batch_v1', 'batch': 2,
        'capacity': 4096, 'prefill_chunk': 16, 'chunk': 16, 'mtp_depth': 3, 'mtp_policy': 'two_real_rows_equal_limits_else_greedy',
        'shared_compressed_weights': True, 'library_sha256': library_sha,
        'research_only': True, 'precision': 'mixed NVFP4/FP8; BF16 vision'})
    for frame in requests():
        jobs = frame['batch']
        assert isinstance(jobs, list) and 1 <= len(jobs) <= 2
        assert len({job['id'] for job in jobs}) == len(jobs)
        started = time.monotonic()
        speculative.reset()
        rows = []
        for row, job in enumerate(jobs):
            body = job['body']
            assert body['temperature'] == 0 and body['chat_template_kwargs'] == {'enable_thinking': False}
            assert body['model'] == 'qwen38-27b-nvfp4'
            assert len(body['messages']) == 1 and body['messages'][0]['role'] == 'user'
            limit = body['max_tokens']
            assert type(limit) is int and 2 <= limit <= 1024
            content = body['messages'][0]['content']
            visual = None
            image_start = -1
            image_hash = None
            if isinstance(content, str):
                ids = tokenizer.encode('<|im_start|>user\n' + content + suffix)
                positions = [[i] * 3 for i in range(len(ids))]
                current = len(ids)
            else:
                assert len(content) == 2 and content[0]['type'] == 'image_url' and content[1]['type'] == 'text'
                url = content[0]['image_url']['url']
                prefix = 'data:image/jpeg;base64,'
                assert url.startswith(prefix)
                data = base64.b64decode(url[len(prefix):], validate=True)
                assert data[:2] == b'\xff\xd8'
                image_hash = hashlib.sha256(data).hexdigest()
                image = root / 'tmp' / ('batch-stream-row-' + str(row) + '.jpg')
                image.write_bytes(data)
                prepared = prepare_cuda_image(io, checkpoint, str(image))
                assert [prepared[1], prepared[2]] == [16, 16]
                visual = io.clone(vision.encode_patches(prepared[0]))
                leading = tokenizer.encode('<|im_start|>user\n<|vision_start|>')
                image_start = len(leading)
                ids = leading + [config['image_token_id']] * 64 + tokenizer.encode('<|vision_end|>' + content[1]['text'] + suffix)
                positions, current = image_positions(len(ids), image_start, 64, 8)
            assert 0 < len(ids) and len(ids) + limit <= 4096
            embeddings = io.embedding(ids)
            hidden = [io.visual_token(visual, i-image_start) if visual is not None and image_start <= i < image_start+64
                else io.row(embeddings, i) for i in range(len(ids))]
            rows.append({'job': job, 'ids': ids, 'positions': positions, 'current': current,
                'limit': limit, 'hidden': hidden, 'embeddings': embeddings, 'visual': visual,
                'image_hash': image_hash, 'generated': [], 'text': ''})
        # The physical profile rejects zero-length prefill. A missing row borrows
        # the first valid prompt; it is inactive for generation and emits nothing.
        physical_rows = rows if len(rows) == 2 else rows + [rows[0]]
        count = max(len(r['ids']) for r in physical_rows)
        zero = io.zeros([1, config['text_config']['hidden_size']])
        hidden = [io.stack_rows([r['hidden'][t] if t < len(r['ids']) else zero for r in physical_rows]) for t in range(count)]
        positions = [r['positions'] + [[0, 0, 0]] * (count-len(r['ids'])) for r in physical_rows]
        # Keep dummy rows and unequal limits on the previously qualified greedy path.
        if len(rows) == 2 and rows[0]['limit'] == rows[1]['limit']:
            def notify(row, token, sampled):
                r = rows[row]
                if sampled == 1 and len(r['generated']) == 1:
                    assert r['generated'][0] == token
                    return
                assert sampled == len(r['generated']) + 1
                now = time.monotonic()
                r['generated'].append(token)
                r['text'] = tokenizer.decode([t for t in r['generated'] if t not in stops])
                emit({'event': 'token', 'id': r['job']['id'], 'token_id': token,
                    'sampled_tokens': sampled, 'prompt_tokens': len(r['ids']), 'text': r['text'],
                    'stop': token in stops, 'native_monotonic_seconds': now})
            def first(prediction):
                for row, token in enumerate(prediction):
                    notify(row, token, 1)
            speculative.prefill(hidden, positions, [len(r['ids']) for r in rows],
                [r['current'] for r in rows], on_first_token=first)
            result = speculative.generate(rows[0]['limit'], on_token=notify)
            for row, r in enumerate(rows):
                expected = list(result['generated_token_ids'][row])
                if result['stop_token_ids'][row] is not None:
                    expected.append(result['stop_token_ids'][row])
                assert expected == r['generated']
                assert result['sampled_tokens_including_stop'][row] == len(expected)
                emit({'event': 'completed', 'id': r['job']['id'], 'generated_ids': r['generated'],
                    'prompt_ids': r['ids'], 'text': r['text'], 'image_sha256': r['image_hash'],
                    'generation_mode': 'verified_mtp', 'speculative_metrics': result['speculative_metrics'],
                    'finish_reason': 'stop' if r['generated'][-1] in stops else 'length',
                    'native_elapsed_seconds': time.monotonic()-started})
            continue
        prediction = decoder.prefill_sequence(hidden, positions, [len(r['ids']) for r in physical_rows])
        active = [1, int(len(rows) == 2)]
        current = [r['current'] for r in physical_rows]
        for step in range(max(r['limit'] for r in rows)):
            for row, r in enumerate(rows):
                if not active[row]:
                    continue
                token = prediction[row]
                now = time.monotonic()
                r['generated'].append(token)
                r['text'] = tokenizer.decode([t for t in r['generated'] if t not in stops])
                emit({'event': 'token', 'id': r['job']['id'], 'token_id': token,
                    'sampled_tokens': len(r['generated']), 'prompt_tokens': len(r['ids']),
                    'text': r['text'], 'stop': token in stops, 'native_monotonic_seconds': now})
                if token in stops or len(r['generated']) == r['limit']:
                    active[row] = 0
                    emit({'event': 'completed', 'id': r['job']['id'], 'generated_ids': r['generated'],
                        'prompt_ids': r['ids'], 'text': r['text'], 'image_sha256': r['image_hash'],
                        'generation_mode': 'greedy_target', 'finish_reason': 'stop' if token in stops else 'length',
                        'native_elapsed_seconds': time.monotonic()-started})
            if not any(active):
                break
            prediction = decoder.step(prediction, active, [[p]*3 for p in current])
            current = [p+flag for p, flag in zip(current, active)]
finally:
    if speculative is not None:
        speculative.close()
    if vision is not None:
        vision.close()
    if decoder is not None:
        decoder.close()
