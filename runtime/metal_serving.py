# SPDX-License-Identifier: Apache-2.0
"""Galaxy serving contract backed by Garnet's native Metal graph executors."""
import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.dirname(os.path.dirname(__file__))
# The native package registers import roots with XLang3. Embedded Cantor
# runtimes need not expose CPython's sys.path.
import garnet
from xModel.qwen.metal_dense.decoder import DenseMetalDecoder
from xModel.qwen35.vl_0_8b.metal_decoder import MetalDecoder
from xModel.qwen35.vl_0_8b.metal_vision import MetalImageEncoder

_models = {}
_lock = threading.RLock()


class Model:
    def __init__(self, root, cache, profile, name):
        self.root, self.name = root, name
        self.cache = cache or os.path.join(root, 'metal-cache')
        os.makedirs(self.cache, exist_ok=True)
        with open(os.path.join(root, 'config.json')) as stream:
            self.config = json.loads(stream.read())
        self.visual = 'vision_config' in self.config
        self.text_config = self.config['text_config'] if self.visual else self.config
        self.capacity = int(profile.get('maxInputTokens', 1536)) + int(profile.get('maxOutputTokens', 512))
        if self.capacity < 64 or self.capacity > 8192:
            raise ValueError('Metal serving capacity must be between 64 and 8192 tokens')
        self.tokenizer = garnet.tokenizer()
        self.tokenizer.open(root)
        checkpoint = os.path.join(root, 'model.safetensors')
        cls = MetalDecoder if self.visual else DenseMetalDecoder
        self.decoder = cls(checkpoint, self.text_config, 1, self.capacity)
        self.decoder.enable_prefill(16, 'shared')
        self.vision, self.grid = None, None
        self.stops = [self.tokenizer.token_id('<|im_end|>'), self.tokenizer.token_id('<|endoftext|>')]
        eos = self.text_config.get('eos_token_id')
        if isinstance(eos, list):
            self.stops.extend(eos)
        elif eos is not None:
            self.stops.append(eos)

    def infer(self, prompt, image, limit):
        start = time.monotonic()
        hidden, positions = [], []
        current = 0
        self.decoder.reset()
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
            python = os.environ.get('GARNET_METAL_FRONTEND_PYTHON', '')
            if not python or not os.path.isfile(python):
                raise RuntimeError('Set GARNET_METAL_FRONTEND_PYTHON to a Python with NumPy and Pillow')
            stem = os.path.join(self.cache, 'request-frame')
            subprocess.run([python, os.path.join(ROOT, 'runtime', 'metal_prepare_image.py'),
                            self.root, image, stem], check=True, timeout=60)
            with open(stem + '.json') as stream:
                metadata = json.loads(stream.read())
            gh, gw = metadata['grid']
            if self.grid != [gh, gw]:
                if self.vision is not None:
                    self.vision.close()
                self.vision = MetalImageEncoder(os.path.join(self.root, 'model.safetensors'),
                                                self.config['vision_config'], gh, gw, 1)
                self.grid = [gh, gw]
            embedding = self.vision.encode_file(stem + '.f32')
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
        prediction = self.decoder.prefill_sequence(hidden, [positions])
        first = time.monotonic()
        output = []
        for index in range(limit):
            token = prediction[0]
            if token in self.stops:
                break
            output.append(token)
            if index + 1 == limit:
                break
            prediction = self.decoder.step(prediction, active=[1], positions=[[current, current, current]])
            current += 1
        return dict(status='ok', text=self.tokenizer.decode(output), model_id=self.name,
                    backend=garnet.backend_info(), generated_tokens=len(output),
                    ttft_seconds=first-start, total_seconds=time.monotonic()-start)


def serve_model(root, xmodel, cache, profile, name):
    with _lock:
        try:
            if name not in _models:
                _models[name] = Model(root, cache, json.loads(profile) if profile else {}, name)
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
