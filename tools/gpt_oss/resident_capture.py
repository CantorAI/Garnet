"""Bounded Nsight captures on trial zero; never an unprofiled benchmark."""
import math


def existing_capture_artifacts(prefix):
    from pathlib import Path
    prefix = Path(prefix)
    # The invoking shell may open PREFIX.log before this controller starts.
    # Keep overwrite protection on result/manifest and Nsight artifacts.
    return [path for path in prefix.parent.glob(prefix.name + '*')
            if path.suffix in ('.json', '.nsys-rep', '.sqlite', '.qdstrm')]


class ResidentCapture:
    def __init__(self, environment, input_tokens, chunk, output_tokens, profiler=None):
        if min(input_tokens, chunk, output_tokens) <= 0:
            raise ValueError('Invalid capture workload')
        prefill_text = environment.get('GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS', '')
        decode_text = environment.get('GARNET_GPT_OSS_PROFILE_DECODE_RANGES', '')
        prefill = [int(value) for value in prefill_text.split(',')] if prefill_text else []
        decode = []
        if decode_text:
            for value in decode_text.split(','):
                parts = value.split(':')
                if len(parts) != 2:
                    raise ValueError('Decode capture must be START:STEPS')
                decode.append(tuple(map(int, parts)))
        if len(set(prefill)) != len(prefill) or any(
                not 0 <= value < math.ceil(input_tokens / chunk) for value in prefill):
            raise ValueError('Duplicate or out-of-bounds prefill capture')
        decode.sort()
        previous_end = 1
        for start, steps in decode:
            if start < previous_end or steps <= 0 or start + steps > output_tokens:
                raise ValueError('Overlapping or out-of-bounds decode capture')
            previous_end = start + steps
        self.prefill = sorted(prefill)
        self.decode = decode
        self.requested = [('prefill', index, 1) for index in self.prefill] + [
            ('decode', start, steps) for start, steps in decode]
        self.enabled = bool(self.requested)
        diagnostic = environment.get('GARNET_BENCH_RESIDENT_DIAGNOSTIC', '0')
        if diagnostic not in ('0', '1') or (diagnostic == '1') != self.enabled:
            raise ValueError('Resident profiling requires explicit diagnostic selection')
        if any(environment.get(key, '0') != '0' for key in (
                'GARNET_GPT_OSS_PROFILE_PREFILL', 'GARNET_GPT_OSS_PROFILE_DECODE_STEPS')):
            raise ValueError('Legacy profiling controls cannot select resident capture')
        if self.enabled and (not environment.get('GARNET_BENCH_NSYS_OUTPUT') or
                not 1 <= len(self.requested) <= 8 or
                environment.get('GARNET_BENCH_CAPTURE_RANGES') != str(len(self.requested))):
            raise ValueError('Resident diagnostic requires Nsight output and exact range count')
        self.profiler = profiler
        if self.enabled and profiler is None:
            import ctypes
            self.profiler = ctypes.CDLL('libcudart.so')
        self.active = None
        self.completed = []

    def _begin(self, selection):
        if self.active is not None or selection in self.completed:
            raise RuntimeError('Invalid repeated/overlapping capture execution')
        if self.profiler.cudaProfilerStart() != 0:
            raise RuntimeError('cudaProfilerStart failed')
        self.active = selection

    def _end(self):
        selection = self.active
        if self.profiler.cudaProfilerStop() != 0:
            raise RuntimeError('cudaProfilerStop failed')
        self.active = None
        self.completed.append(selection)

    def prefill_begin(self, trial, index):
        if trial == 0 and index in self.prefill:
            self._begin(('prefill', index, 1))

    def prefill_end(self, trial, index):
        if trial == 0 and self.active == ('prefill', index, 1):
            self._end()

    def decode_begin(self, trial, index):
        if trial == 0:
            for start, steps in self.decode:
                if index == start:
                    self._begin(('decode', start, steps))

    def decode_end(self, trial, index):
        if trial == 0 and self.active is not None:
            phase, start, steps = self.active
            if phase == 'decode' and index == start + steps - 1:
                self._end()

    def abort(self):
        # Close the tool range on application failure without claiming it
        # completed. The caller still propagates the original failure.
        if self.active is not None:
            if self.profiler.cudaProfilerStop() != 0:
                raise RuntimeError('cudaProfilerStop failed during cleanup')
            self.active = None

    def assert_complete(self):
        if self.active is not None or self.completed != self.requested:
            raise RuntimeError('Resident diagnostic did not complete every capture')

    def metadata(self):
        return dict(profiled_diagnostic=self.enabled,
            profile_prefill=bool(self.prefill), profile_prefill_chunks=self.prefill,
            profile_decode_steps=sum(steps for _, steps in self.decode),
            profile_decode_ranges=self.decode, completed_capture_ranges=self.completed)
