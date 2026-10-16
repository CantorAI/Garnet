from pathlib import Path
s=Path('/Users/shawn/CantorAI/out/research/worldsense-prefix-cache/source');p=s/'runtime/metal_serving.py';t=p.read_text();needle='        self.tokenizer = garnet.tokenizer()';assert needle in t;t=t.replace(needle,'''        self._prefix_supported = not self.visual and self.text_config.get('model_type') == 'qwen3' and self.text_config.get('hidden_size') == 2048 and self.text_config.get('num_hidden_layers') == 28 and 'dense_prefix_cache' in garnet.backend_info()
        self.text_prefix_cache = bool(profile.get('textPrefixCache', False))
        if self.text_prefix_cache and not self._prefix_supported:
            raise ValueError('Native prefix cache requires the validated Qwen3-1.7B profile')
        self._prefix_keys = {}
'''+needle)
needle='    def infer(self, prompt, image, limit):';idx=t.index(needle)
code='''    def _prefill_with_prefix(self, hidden, positions, token_ids, stages):
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

''';t=t[:idx]+code+t[idx:]
t=t.replace('        hidden, positions = [], []','        hidden, positions, token_ids = [], [], []').replace('            for token in self.tokenizer.encode(text):\n','            for token in self.tokenizer.encode(text):\n                token_ids.append(token)\n')
t=t.replace('        prediction = self.decoder.prefill_sequence(hidden, [positions])','        prediction = self._prefill_with_prefix(hidden, positions, token_ids, stages)')
t=t.replace("        stages['decode_fused_head_dispatches'] = fused_head_dispatches","""        stages['decode_fused_head_dispatches'] = fused_head_dispatches
        stages['prefix_cache_native_bytes'] = self.decoder.session.prefix_cache_bytes
        stages['prefix_cache_native_entries'] = self.decoder.session.prefix_cache_entries
        stages['prefix_restore_native_bytes'] = self.decoder.session.prefix_restore_bytes
        stages['prefix_capture_native_bytes'] = self.decoder.session.prefix_capture_bytes
        stages['serving_source'] = __file__""")
t=t.replace('serving_configuration=dict(capacity=self.capacity,','serving_configuration=dict(text_prefix_cache=self.text_prefix_cache, capacity=self.capacity,')
start=t.index('def set_benchmark_prefill_policy(enabled):');t=t[:start]+'''def set_benchmark_prefill_policy(enabled):
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
''';p.write_text(t);print('Serving prefix path written; not executed')
