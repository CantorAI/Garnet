# SPDX-License-Identifier: Apache-2.0
"""Galaxy serving contract backed by Garnet's native Metal graph executors."""
import json
import os
import threading
import hashlib
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
        self.initialization_memory = dict(before=json.loads(garnet.memory_stats_json()), other_model_ids_at_start=list(_models))
        self.collect_inventory = bool(profile.get("collectNativeInventory", False))
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
        self._prefix_supported = not self.visual and self.text_config.get('model_type') == 'qwen3' and self.text_config.get('hidden_size') == 2048 and self.text_config.get('num_hidden_layers') == 28 and 'dense_prefix_cache' in garnet.backend_info()
        self.text_prefix_cache = bool(profile.get('textPrefixCache', False))
        if self.text_prefix_cache and not self._prefix_supported:
            raise ValueError('Native prefix cache requires the validated Qwen3-1.7B profile')
        self._prefix_keys = {}
        self.tokenizer = garnet.tokenizer()
        self.tokenizer.open(root)
        index = os.path.join(root, 'model.safetensors.index.json')
        checkpoint = index if os.path.isfile(index) else os.path.join(root, 'model.safetensors')
        self.checkpoint = checkpoint
        cls = MetalDecoder if self.visual else DenseMetalDecoder
        packed_limit = 64 * 2560 if self.visual and self.xmodel == 'vl_4b' and 'qwen35_4b_wide_prefill' in garnet.backend_info() else 65536
        if self.visual and self.prefill_chunk * self.text_config['hidden_size'] > packed_limit:
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

        self.initialization_memory['after_decoder_ready'] = json.loads(garnet.memory_stats_json())

    def _prefill_with_prefix(self, hidden, positions, token_ids, stages):
        stages['prefix_cache_hit'] = False
        stages['prefix_cache_restored_tokens'] = 0
        stages['prefix_cache_captured_tokens'] = 0
        stages['prefix_cache_restore_seconds'] = 0
        stages['prefix_cache_capture_seconds'] = 0
        stages['executed_prefill_tokens'] = len(hidden)
        if not self.text_prefix_cache or self.visual or len(hidden) <= 80:
            return self.decoder.prefill_sequence(hidden, [positions])
        if not self._prefix_supported or self.decoder.chunk != 16 or len(token_ids) != len(hidden):
            raise ValueError('Prefix policy requires validated dense text and chunk16')
        if any(positions[i] != [i, i, i] for i in range(64)):
            raise ValueError('Prefix positions must be identical canonical text positions')
        key = tuple(token_ids[:64])
        stages['prefix_cache_key_sha256'] = hashlib.sha256(json.dumps(list(key)).encode()).hexdigest()
        if key in self._prefix_keys:
            handle = self._prefix_keys.pop(key)
            began = time.monotonic()
            try:
                restored = self.decoder.session.restore_prefix(handle)
                if restored != 64:
                    raise ValueError('Native prefix length changed')
            except:
                self._prefix_keys.clear()
                self.decoder.session.clear_prefix_cache()
                raise
            stages['prefix_cache_restore_seconds'] = time.monotonic() - began
            self._prefix_keys[key] = handle
            self.decoder.slots = [64]
            stages['prefix_cache_hit'] = True
            stages['prefix_cache_restored_tokens'] = 64
            stages['executed_prefill_tokens'] = len(hidden) - 64
            return self.decoder.prefill_sequence(hidden[64:], [positions[64:]])
        self.decoder.prefill_sequence(hidden[:64], [positions[:64]], predict=False)
        prefix_profile = dict(self.decoder.prefill_profile)
        prefix_plan = list(self.decoder.prefill_chunk_plan)
        prefix_compile = self.decoder.prefill_compile_ms
        prefix_inference = self.decoder.prefill_inference_ms
        began = time.monotonic()
        handle = self.decoder.session.capture_prefix(64)
        stages['prefix_cache_capture_seconds'] = time.monotonic() - began
        stages['prefix_cache_captured_tokens'] = 64
        self._prefix_keys[key] = handle
        while len(self._prefix_keys) > 2:
            self._prefix_keys.pop(next(iter(self._prefix_keys)))
        prediction = self.decoder.prefill_sequence(hidden[64:], [positions[64:]])
        self.decoder.prefill_compile_ms += prefix_compile
        self.decoder.prefill_inference_ms += prefix_inference
        self.decoder.prefill_chunk_plan = prefix_plan + self.decoder.prefill_chunk_plan
        for name, value in prefix_profile.items():
            self.decoder.prefill_profile[name] = self.decoder.prefill_profile.get(name, 0) + value
        return prediction

    def infer(self, prompt, image, limit):
        start = time.monotonic()
        stages = dict(initialization_memory=self.initialization_memory, memory_before_request=json.loads(garnet.memory_stats_json()))
        hidden, positions, token_ids = [], [], []
        current = 0
        self.decoder.reset()
        stages['reset_seconds'] = time.monotonic() - start
        def append_text(text):
            nonlocal current
            for token in self.tokenizer.encode(text):
                token_ids.append(token)
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
        stages['memory_after_input'] = json.loads(garnet.memory_stats_json())
        prefill_start = time.monotonic()
        prediction = self._prefill_with_prefix(hidden, positions, token_ids, stages)
        stages['memory_after_prefill'] = json.loads(garnet.memory_stats_json())
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
            current += 1
        stages['decode_seconds'] = time.monotonic() - first
        stages['memory_after_decode'] = json.loads(garnet.memory_stats_json())
        memory_graphs = [layer['graph'] for layer in self.decoder.layers] + list(self.decoder.prefill_graphs.values())
        stages['resident_avoided_storage_bytes'] = sum(graph.resident_avoided_storage_bytes for graph in memory_graphs)
        stages['lazy_default_retired_bytes'] = sum(graph.lazy_default_retired_bytes for graph in memory_graphs)
        stages['lazy_default_rehydrated_bytes'] = sum(graph.lazy_default_rehydrated_bytes for graph in memory_graphs)
        stages['lazy_default_planned_slots'] = sum(graph.lazy_default_planned_slots for graph in memory_graphs)
        if self.collect_inventory:
            descriptors = [dict(object=layer['graph'], tag='decode.layer.'+str(i)) for i,layer in enumerate(self.decoder.layers)]
            descriptors.append(dict(object=self.decoder.head, tag='head'))
            for key in sorted(self.decoder.prefill_graphs):
                descriptors.append(dict(object=self.decoder.prefill_graphs[key], tag='prefill.'+key))
            if self.vision is not None:
                descriptors.append(dict(object=self.vision.graph, tag='vision'))
            inventory_payload = garnet.memory_inventory_json(descriptors, self.decoder.session)
            inventory = json.loads(inventory_payload)
            inventory_bytes = inventory_payload.encode()
            capture_root = '/Users/shawn/CantorAI/out/research/worldsense-lazy-defaults/captures'
            os.makedirs(capture_root, exist_ok=True)
            self._inventory_capture_counter = getattr(self, '_inventory_capture_counter', 0) + 1
            capture_key = hashlib.sha256((self.name + str(time.time()) + str(self._inventory_capture_counter)).encode()).hexdigest()
            capture_path = os.path.join(capture_root, capture_key + '.json')
            if os.path.exists(capture_path):
                raise ValueError('Diagnostic capture identity collision')
            with open(capture_path + '.tmp', 'w') as stream:
                stream.write(inventory_payload)
            os.replace(capture_path + '.tmp', capture_path)
            stages['memory_inventory'] = dict(path=capture_path, sha256=hashlib.sha256(inventory_bytes).hexdigest(),
                bytes=len(inventory_bytes), model_id=self.name, unique_allocated_bytes=inventory['unique_allocated_bytes'],
                buffer_count=len(inventory['buffers']), reference_count=len(inventory['references']),
                device_snapshot=inventory['device_snapshot'], scope=inventory['scope'])
    
        stages['decode_profile'] = decode_profile
        stages['decode_fused_head_dispatches'] = fused_head_dispatches
        stages['prefix_cache_native_bytes'] = self.decoder.session.prefix_cache_bytes
        stages['prefix_cache_native_entries'] = self.decoder.session.prefix_cache_entries
        stages['prefix_restore_native_bytes'] = self.decoder.session.prefix_restore_bytes
        stages['prefix_capture_native_bytes'] = self.decoder.session.prefix_capture_bytes
        stages['serving_source'] = __file__
        return dict(input_prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(), serving_configuration=dict(text_prefix_cache=self.text_prefix_cache, capacity=self.capacity, image_pixels=self.image_pixels, prefill_chunk=self.decoder.chunk, requested_output_limit=int(limit), share_vision_weights=self.share_vision_weights), image_preprocess_backend='native-cpp' if image else '', image_preprocess_seconds=image_preprocess_seconds if image else 0, status='ok', text=self.tokenizer.decode(output), model_id=self.name,
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
    """Benchmark combined VLM64/text16+prefix policy; validate before mutation."""
    with _lock:
        if 'dense_prefix_cache' not in garnet.backend_info() or not _models:
            return json.dumps(dict(supported=False))
        for model in _models.values():
            if model.visual:
                if model.xmodel != 'vl_4b' or model.text_config['hidden_size'] != 2560 or not model.decoder.linear_chunk:
                    raise ValueError('Prefix experiment requires validated4B VLM')
            elif not model._prefix_supported:
                raise ValueError('Prefix experiment requires validated1.7B reasoner')
        for model in _models.values():
            model.decoder.chunk = (64 if enabled else 16) if model.visual else 16
            model.prefill_chunk = model.decoder.chunk
            model.text_prefix_cache = bool(enabled) if not model.visual else False
        return json.dumps(dict(supported=True, enabled=bool(enabled)))
