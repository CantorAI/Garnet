"""Read-only vLLM worker extension for benchmark memory evidence.

RPCs run outside timed inference. Shared KV views are deduplicated by storage,
so hybrid attention groups do not multiply the physical backing allocation.
"""


def kv_backing_allocations(caches):
    import torch
    allocations = {}

    def visit(value):
        if isinstance(value, torch.Tensor):
            storage = value.untyped_storage()
            if value.device.type == 'cuda':
                key = (str(value.device), storage.data_ptr())
                allocations[key] = storage.nbytes()
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    visit(caches)
    return list(allocations.values())


class GptOssBenchmarkProfile:
    def benchmark_kv_profile(self, batch, input_tokens, output_tokens):
        import torch
        runner = self.model_runner
        config = runner.kv_cache_config
        sizes = kv_backing_allocations(runner.kv_caches)
        if not sizes:
            raise RuntimeError('No CUDA KV backing storage found; profile unavailable')
        layers = []
        for name, spec in runner.get_kv_cache_spec().items():
            heads, head = spec.num_kv_heads, spec.head_size
            head_v = spec.head_size_v
            element_bytes = torch.empty((), dtype=spec.dtype).element_size()
            per_token = heads * (head + head_v) * element_bytes
            processed = input_tokens + output_tokens - 1
            window = getattr(spec, 'sliding_window', None)
            retained = min(processed, window) if window else processed
            layers.append(dict(name=name, kind=type(spec).__name__,
                               dtype=str(spec.dtype), block_tokens=spec.block_size,
                               page_bytes=spec.page_size_bytes,
                               sliding_window=window, bytes_per_token=per_token,
                               full_history_data_bytes=batch * processed * per_token,
                               retained_history_data_bytes=batch * retained * per_token))
        return dict(rank=self.rank, device=str(runner.device),
                    allocated_unique_backing_bytes=sum(sizes),
                    backing_allocation_bytes=sizes,
                    num_scheduler_blocks=config.num_blocks,
                    layout=config.kv_cache_layout, layers=layers,
                    logical_processed_tokens_per_request=input_tokens + output_tokens - 1,
                    logical_full_history_bytes=sum(x['full_history_data_bytes'] for x in layers),
                    logical_retained_history_bytes=sum(x['retained_history_data_bytes'] for x in layers),
                    pytorch_peak_allocated_bytes=torch.cuda.max_memory_allocated(runner.device),
                    pytorch_peak_reserved_bytes=torch.cuda.max_memory_reserved(runner.device),
                    occupancy_limit='Logical history at the last decode step, not sampled scheduler block occupancy. Retained bytes exclude page rounding, in-flight tokens and allocator overhead; completed requests are freed.')
