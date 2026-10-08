"""Recorded allocator/session policy at the actual diagnostic subprocess boundary."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import tempfile
from unittest.mock import patch

repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo / 'tools/gpt_oss'))


class ObservedCommand(Exception):
    pass


with tempfile.TemporaryDirectory() as temporary:
    root = Path(temporary)
    request, expected, measured, reference = [root / (name + '.json')
        for name in ('request', 'expected', 'measured', 'reference')]
    request.write_text(json.dumps({'input_ids': list(range(8))}))
    expected.write_text('{}')
    measured.write_text('{}')
    matrix = [list(range(16)), list(range(16))]
    record = dict(batch=2, input_token_ids=list(range(8)),
        output_tokens_per_request=16, max_context_tokens_per_request=32,
        prefill_chunk_tokens=4, complete_warmup_count=1,
        resident_profile=str(measured),
        resident_profile_sha256=hashlib.sha256(measured.read_bytes()).hexdigest(),
        decode_trials=[{'token_ids_by_request': matrix} for _ in range(3)])

    def execute(label, policy, wanted, reject=False):
        reference.write_text(json.dumps(dict(record,
            optimization_environment=policy)))
        prefix = root / label
        observed = []

        def stop(command, **kwargs):
            assert Path(command[1]).name == 'benchmark_batch_tp2.sh'
            environment = kwargs['env']
            assert environment['GARNET_TRT_SYNC_ALLOCATOR'] == wanted
            assert 'GARNET_RESIDENT_SESSION' not in environment
            assert environment['GARNET_RESIDENT_WARMUPS'] == '1'
            assert environment['GARNET_RESIDENT_PROFILE'] == str(measured.resolve())
            assert environment['GARNET_GPT_OSS_MARLIN_FAST_HOST_PACK'] == '1'
            assert environment['GARNET_BENCH_RESIDENT_DIAGNOSTIC'] == '1'
            observed.append(command)
            raise ObservedCommand

        with patch.dict(os.environ, {
                'GARNET_TRT_SYNC_ALLOCATOR': 'inherited-invalid',
                'GARNET_RESIDENT_SESSION': '1',
                'GARNET_RESIDENT_WARMUPS': '99'}), \
             patch.object(sys, 'argv', ['profile_resident_tp2.py',
                 str(request), str(expected), str(reference), str(prefix)]), \
             patch('subprocess.run', side_effect=stop):
            try:
                runpy.run_path(str(repo / 'tools/gpt_oss/profile_resident_tp2.py'),
                    run_name='__main__')
            except ObservedCommand:
                assert not reject
            except ValueError:
                assert reject
            else:
                raise AssertionError('Expected an actual subprocess boundary or rejection')
        if reject:
            assert not observed and not prefix.with_suffix('.capture-manifest.json').exists()
        else:
            assert len(observed) == 1
            manifest = json.loads(Path(str(prefix) + '.capture-manifest.json').read_text())
            assert manifest['status'] == 'failed'  # Deliberately stopped; no GPU result.
            assert manifest['source_reference_session'] == policy.get('GARNET_RESIDENT_SESSION', '0')
            assert manifest['diagnostic_session_mode'] is False
            assert manifest['synchronous_allocator'] == wanted

    common = {'GARNET_GPT_OSS_MARLIN_FAST_HOST_PACK': '1',
              'GARNET_RESIDENT_SESSION': '1'}
    for allocator in ('0', '1'):
        execute('allocator' + allocator,
            dict(common, GARNET_TRT_SYNC_ALLOCATOR=allocator), allocator)
    execute('legacy-default', common, '0')
    for index, invalid in enumerate(('2', 'true', '', 1)):
        execute('invalid' + str(index),
            dict(common, GARNET_TRT_SYNC_ALLOCATOR=invalid), None, reject=True)

print('Recorded allocator/default, inherited-policy clearing, single-case capture and invalid-policy boundary gates PASS')
