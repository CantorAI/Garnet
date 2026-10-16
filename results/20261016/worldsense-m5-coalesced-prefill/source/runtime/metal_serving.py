# SPDX-License-Identifier: Apache-2.0
"""Galaxy serving contract backed by Garnet's native Metal graph executors."""
import json
import os
import threading
import time

# The native package registers import roots with XLang3. Embedded Cantor
# runtimes need not expose CPython's sys.path.
import garnet
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder

_models = {}
_lock = threading.RLock()


class Model:
    def __init__(self, root, xmodel, cache, profile, name):
        self.root, self.name = root, name
        self.cache = cache or os.path.join(root, 'metal-cache')
        os.makedirs(self.cache, exist_ok=True)
        with open(os.path.join(root, 'config.json')) as stream:
            self.config = json.loads(stream.read())
        self.visual = 'vision_config' in self.config
        self.xmodel = os.path.basename(os.path.normpath(xmodel)) if xmodel else ''
        self.image_pixels = int(profile.get('imagePixels', profile.get('maxPixels', 65536)))
        if self.image_pixels < 4096 or self.image_pixels > 1048576:
            raise ValueError('Metal imagePixels must be between 4096 and 1048576')
        self.prefill_chunk = int(profile.get('prefillChunk', 16))
        if self.prefill_chunk not in (2, 4, 8, 16, 32, 64):
            raise ValueError('Unsupported Metal prefillChunk')
        self.share_vision_weights = bool(profile.get('shareVisionWeights', self.xmodel == 'vl_4b'))
        if self.visual:
            with open(os.path.join(root, 'preprocessor_config.json')) as stream:
                processor = json.loads(stream.read())
            v = self.config['vision_config']
            self.image_options = json.dumps(dict(patch_size=v['patch_size'], merge_size=v['spatial_merge_size'],
                temporal_patch_size=v['temporal_patch_size'], pixels=self.image_pixels,
                image_mean=processor['image_mean'], image_std=processor['image_std']))
        self.text_config = self.config['text_config'] if self.visual else self.config
        self.capacity = int(profile.get('maxInputTokens', 1536)) + int(profile.get('maxOutputTokens', 512))
        if self.capacity < 64 or self.capacity > 8192:
            raise ValueError('Metal serving capacity must be between 64 and 8192 tokens')
        self.tokenizer = garnet.tokenizer()
        self.tokenizer.open(root)
        index = os.path.join(root, 'model.safetensors.index.json')
        checkpoint = index if os.path.isfile(index) else os.path.join(root, 'model.safetensors')
        self.checkpoint = checkpoint
        cls = MetalDecoder if self.visual else DenseMetalDecoder
        if self.visual and self.prefill_chunk * self.text_config['hidden_size'] > 65536:
            raise ValueError('Metal prefill chunk times hidden size exceeds the packed width limit')
        self.model_factory = None
        if self.visual and self.xmodel == 'vl_4b':
            from xModel.qwen35.vl_4b import model
            self.model_factory = model
            self.decoder = model.create_decoder(checkpoint, self.config, self.capacity)
        else:
            if self.visual and self.text_config.get('hidden_size') == 2560:
                raise ValueError('Qwen3.5-4B requires xmodelRoot xModel/qwen35/vl_4b')
            self.decoder = cls(checkpoint, self.text_config, 1, self.capacity)
        if self.visual:
            self.decoder.enable_prefill(self.prefill_chunk, 'shared', bool(profile.get('linearChunk', False)))
        else:
            self.decoder.enable_prefill(self.prefill_chunk, 'shared')
        self.vision, self.grid = None, None
        self.stops = [self.tokenizer.token_id('<|im_end|>'), self.tokenizer.token_id('<|endoftext|>')]
        eos = self.text_config.get('eos_token_id')
        if isinstance(eos, list):
            self.stops.extend(eos)
        elif eos is not None:
            self.stops.append(eos)

    def infer(self, prompt, image, limit):
        start = time.monotonic()
        stages = {'serving_source': __file__}
        hidden, positions = [], []
        current = 0
        self.decoder.reset()
        stages['reset_seconds'] = time.monotonic() - start
        def append_text(text):
            nonlocal current
            for token in self.tokenizer.encode(text):
                hidden.append(self.decoder.io.embedding([token]))
                positions.append([current, current, current])
                current += 1
        append_text('<|im_start|>user\n')
        if image:
            if not self.visual:
                raise ValueError('The selected model does not accept images')
            image_start = time.monotonic()
            prepared = garnet.prepare_image(image, self.image_options)
            image_preprocess_seconds = time.monotonic()-image_start
            gh, gw = prepared[1], prepared[2]
            vision_start = time.monotonic()
            if self.grid != [gh, gw]:
                # Keep the current graph usable if construction fails. Shared
                # native weights own their buffers after the old graph closes.
                previous = self.vision
                shared = previous if self.share_vision_weights else None
                replacement = self.model_factory.create_vision(self.checkpoint, self.config, gh, gw, shared) if self.model_factory else MetalImageEncoder(self.checkpoint, self.config['vision_config'], gh, gw, 1, shared)
                self.vision, self.grid = replacement, [gh, gw]
                if previous is not None:
                    previous.close()
            embedding = self.vision.encode_patches(prepared[0])
            stages['vision_seconds'] = time.monotonic() - vision_start
            stages['image_grid'] = [gh, gw]
            append_text('<|vision_start|>')
            merge = self.config['vision_config']['spatial_merge_size']
            mh, mw = gh // merge, gw // merge
            image_base = current
            for index in range(mh * mw):
                hidden.append(self.decoder.io.visual_token(embedding, index))
                positions.append([image_base, image_base + index // mw, image_base + index % mw])
            current = image_base + max(mh, mw)
            append_text('<|vision_end|>')
        append_text(prompt + '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n')
        if len(hidden) + limit > self.capacity:
            raise ValueError('Prompt and requested output exceed the admitted Metal capacity')
        prefill_start = time.monotonic()
        prediction = self.decoder.prefill_sequence(hidden, [positions])
        first = time.monotonic()
        stages['prefill_seconds'] = first - prefill_start
        stages['prefill_compile_ms'] = self.decoder.prefill_compile_ms
        stages['prefill_profile'] = self.decoder.prefill_profile
        stages['input_tokens'] = len(hidden)
        stages['prefill_chunk_plan'] = self.decoder.prefill_chunk_plan
        stages['host_input_seconds'] = prefill_start - start - stages['reset_seconds'] - stages.get('vision_seconds', 0) - (image_preprocess_seconds if image else 0)
        output = []
        decode_profile = dict(prepare_ms=0, encode_ms=0, gpu_ms=0, wait_ms=0)
        fused_head_dispatches = 0
        coalesced_prefill_decode_dispatches = 0
        for index in range(limit):
            token = prediction[0]
            if token in self.stops:
                break
            output.append(token)
            if index + 1 == limit:
                break
            prediction = self.decoder.step(prediction, active=[1], positions=[[current, current, current]])
            for key in decode_profile:
                decode_profile[key] += getattr(self.decoder.session, key)
            fused_head_dispatches += self.decoder.session.fused_head_argmax_dispatches
            coalesced_prefill_decode_dispatches += getattr(self.decoder.session, 'coalesced_prefill_dispatches', 0)
            current += 1
        stages['decode_seconds'] = time.monotonic() - first
        stages['decode_profile'] = decode_profile
        stages['decode_fused_head_dispatches'] = fused_head_dispatches
        stages['decode_coalesced_prefill_dispatches'] = coalesced_prefill_decode_dispatches
        return dict(image_preprocess_backend='native-cpp' if image else '', image_preprocess_seconds=image_preprocess_seconds if image else 0, status='ok', text=self.tokenizer.decode(output), model_id=self.name,
                    backend=garnet.backend_info(), generated_tokens=len(output),
                    timings=stages, generated_token_ids=output,
                    ttft_seconds=first-start, total_seconds=time.monotonic()-start)


def serve_model(root, xmodel, cache, profile, name):
    with _lock:
        try:
            if name not in _models:
                _models[name] = Model(root, xmodel, cache, json.loads(profile) if profile else {}, name)
            return serve_status_json(name)
        except Exception as error:
            return json.dumps(dict(ready=False, model_id=name, error_message=str(error)))


def serve_status_json(name):
    return json.dumps(dict(ready=name in _models, state='ready' if name in _models else 'unloaded', model_id=name))


def infer_json(prompt, image, limit, name):
    with _lock:
        try:
            if name not in _models:
                raise ValueError('Model is not loaded: ' + name)
            return json.dumps(_models[name].infer(prompt, image, int(limit)))
        except Exception as error:
            return json.dumps(dict(status='error', error_message=str(error), model_id=name))


def set_benchmark_prefill_policy(enabled):
    """Synchronous benchmark ablation; no model state or weights are changed."""
    with _lock:
        if 'coalesced_prefill' not in garnet.backend_info() or not _models:
            return json.dumps(dict(supported=False))
        for model in _models.values():
            if not model.decoder.session.set_coalesced_prefill(bool(enabled)):
                raise ValueError('Native prefill policy is unavailable')
        return json.dumps(dict(supported=True, enabled=bool(enabled)))
