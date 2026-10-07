"""CPU-only independent capture boundary/failure and workload guards."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'tools/gpt_oss'))
from resident_capture import ResidentCapture, existing_capture_artifacts
import tempfile

with tempfile.TemporaryDirectory() as directory:
    prefix = Path(directory) / 'capture'
    prefix.with_suffix('.log').write_text('external shell log')
    assert not existing_capture_artifacts(prefix)
    for suffix in ('.json', '.capture-manifest.json', '_1.nsys-rep', '.sqlite', '.qdstrm'):
        artifact = Path(str(prefix) + suffix)
        artifact.write_text('existing evidence')
        assert artifact in existing_capture_artifacts(prefix)
        artifact.unlink()


class Profiler:
    def __init__(self):
        self.events = []
        self.position = None
        self.start_error = self.stop_error = 0

    def cudaProfilerStart(self):
        self.events.append((self.position, 'start'))
        return self.start_error

    def cudaProfilerStop(self):
        self.events.append((self.position, 'stop'))
        return self.stop_error


env = dict(GARNET_BENCH_RESIDENT_DIAGNOSTIC='1', GARNET_BENCH_NSYS_OUTPUT='diagnostic',
    GARNET_BENCH_CAPTURE_RANGES='4', GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS='0,2',
    GARNET_GPT_OSS_PROFILE_DECODE_RANGES='1:2,9:2')
profiler = Profiler()
capture = ResidentCapture(env, 9, 4, 12, profiler)
for trial in (-1, 0, 1, 2):
    for index in range(3):
        profiler.position = (trial, 'prefill', index)
        capture.prefill_begin(trial, index)
        capture.prefill_end(trial, index)
    for index in range(1, 12):
        profiler.position = (trial, 'decode', index)
        capture.decode_begin(trial, index)
        capture.decode_end(trial, index)
capture.assert_complete()
assert profiler.events == [
    ((0, 'prefill', 0), 'start'), ((0, 'prefill', 0), 'stop'),
    ((0, 'prefill', 2), 'start'), ((0, 'prefill', 2), 'stop'),
    ((0, 'decode', 1), 'start'), ((0, 'decode', 2), 'stop'),
    ((0, 'decode', 9), 'start'), ((0, 'decode', 10), 'stop')]
assert capture.metadata()['profile_decode_steps'] == 4
assert capture.metadata()['profiled_diagnostic']


def rejected(environment, input_tokens=9, chunk=4, output=12):
    try:
        ResidentCapture(environment, input_tokens, chunk, output, Profiler())
    except ValueError:
        return
    raise AssertionError('Unsafe capture selection accepted')


for changes in (
    {'GARNET_BENCH_RESIDENT_DIAGNOSTIC': '0'},
    {'GARNET_BENCH_RESIDENT_DIAGNOSTIC': 'yes'},
    {'GARNET_BENCH_NSYS_OUTPUT': ''},
    {'GARNET_BENCH_CAPTURE_RANGES': '3'},
    {'GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS': '0,0'},
    {'GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS': '-1,2'},
    {'GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS': '0,3'},
    {'GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS': '0,'},
    {'GARNET_GPT_OSS_PROFILE_DECODE_RANGES': '1:2,2:3'},
    {'GARNET_GPT_OSS_PROFILE_DECODE_RANGES': '0:1,9:2'},
    {'GARNET_GPT_OSS_PROFILE_DECODE_RANGES': '1:0,9:2'},
    {'GARNET_GPT_OSS_PROFILE_DECODE_RANGES': '1:2,11:2'},
    {'GARNET_GPT_OSS_PROFILE_DECODE_RANGES': '1:2:3'},
    {'GARNET_GPT_OSS_PROFILE_DECODE_RANGES': '1:2,'},
    {'GARNET_GPT_OSS_PROFILE_PREFILL': '1'},
    {'GARNET_GPT_OSS_PROFILE_DECODE_STEPS': '3'},
    {'GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS': '',
     'GARNET_GPT_OSS_PROFILE_DECODE_RANGES': ''},
):
    rejected(dict(env, **changes))
rejected(dict(env, GARNET_BENCH_CAPTURE_RANGES='9',
    GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS=','.join(map(str, range(9))),
    GARNET_GPT_OSS_PROFILE_DECODE_RANGES=''), input_tokens=36)
rejected(env, input_tokens=0)

# Real2005 input, chunk32: index62 is the padded tail,63 is invalid.
tail = dict(GARNET_BENCH_RESIDENT_DIAGNOSTIC='1', GARNET_BENCH_NSYS_OUTPUT='tail',
    GARNET_BENCH_CAPTURE_RANGES='1', GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS='62')
ResidentCapture(tail, 2005, 32, 512, Profiler())
rejected(dict(tail, GARNET_GPT_OSS_PROFILE_PREFILL_CHUNKS='63'), 2005, 32, 512)

disabled = ResidentCapture({}, 9, 4, 12, profiler)
disabled.assert_complete()
assert not disabled.metadata()['profiled_diagnostic']

def incomplete(value):
    try:
        value.assert_complete()
    except RuntimeError:
        return
    raise AssertionError('Incomplete capture accepted')

profiler = Profiler()
capture = ResidentCapture(tail, 2005, 32, 512, profiler)
incomplete(capture)
profiler.start_error = 1
try:
    capture.prefill_begin(0, 62)
except RuntimeError:
    pass
else:
    raise AssertionError('Start failure ignored')
assert capture.active is None
profiler.start_error = 0
capture.prefill_begin(0, 62)
capture.abort()
assert capture.completed == []
incomplete(capture)

capture = ResidentCapture(tail, 2005, 32, 512, profiler)
capture.prefill_begin(0, 62)
profiler.stop_error = 1
try:
    capture.prefill_end(0, 62)
except RuntimeError:
    pass
else:
    raise AssertionError('Stop failure ignored')
assert capture.active is not None and not capture.completed
profiler.stop_error = 0
capture.abort()
incomplete(capture)
print('Resident capture boundaries, warmup exclusion, workload guards and API failure cleanup PASS')
