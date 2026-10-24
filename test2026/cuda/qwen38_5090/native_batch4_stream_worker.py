"""Research B4 T8 shared-weight streaming worker; GPU/HTTP gates precede timings."""
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
proof = json.loads((root / 'results/shared-full-b4-t8-attempt1-verified.json').read_text())
library_sha = hashlib.sha256((root / 'build/bin/garnet.dll').read_bytes()).hexdigest()
assert proof['status'] == 'shared_full_b4_t8_id_gate_verified' and proof['model_gate_passed']
assert proof['library_sha256'] == library_sha
validate_cuda_batch_study_profile(config, 4096, 4)
tokenizer = garnet.tokenizer()
tokenizer.open(checkpoint)
stops = [tokenizer.token_id('<|im_end|>'), tokenizer.token_id('<|endoftext|>')]
suffix = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
decoder = None
vision = None
try:
    decoder = CUDAPackedDecoder(checkpoint, config['text_config'], 4, 4096, 8, 8,
        flat_concat=True, external_compressed_weights=True)
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
    emit({'event': 'ready', 'protocol': 'study_batch_v1', 'batch': 4,
        'capacity': 4096, 'prefill_chunk': 8, 'chunk': 8, 'mtp_depth': 0,
        'shared_compressed_weights': True, 'library_sha256': library_sha,
        'research_only': True, 'precision': 'mixed NVFP4/FP8; BF16 vision'})
    for frame in requests():
        jobs = frame['batch']
        assert isinstance(jobs, list) and 1 <= len(jobs) <= 4
        assert len({job['id'] for job in jobs}) == len(jobs)
        started = time.monotonic()
        decoder.reset()
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
        physical_rows = rows + [rows[0]] * (4 - len(rows))
        count = max(len(r['ids']) for r in physical_rows)
        zero = io.zeros([1, config['text_config']['hidden_size']])
        hidden = [io.stack_rows([r['hidden'][t] if t < len(r['ids']) else zero for r in physical_rows]) for t in range(count)]
        positions = [r['positions'] + [[0, 0, 0]] * (count-len(r['ids'])) for r in physical_rows]
        prediction = decoder.prefill_sequence(hidden, positions, [len(r['ids']) for r in physical_rows])
        active = [1] * len(rows) + [0] * (4 - len(rows))
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
    if vision is not None:
        vision.close()
    if decoder is not None:
        decoder.close()
