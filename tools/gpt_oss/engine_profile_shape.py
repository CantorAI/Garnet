"""Unvalidated engine-profile shapes derived from immutable completed evidence."""
import hashlib
import json


def candidate_shape(reference, reference_bytes, *, batch=None, output=None,
                    context=None, prefill_chunk=None):
    ids = reference.get('input_token_ids')
    if (not isinstance(ids, list) or not ids or
            any(type(token) is not int or token < 0 for token in ids) or
            type(reference.get('input_tokens_per_request')) is not int or
            reference['input_tokens_per_request'] != len(ids)):
        raise ValueError('New shapes require verified complete original input IDs')
    selected = dict(batch=reference['batch'] if batch is None else batch,
        output_tokens_per_request=reference['output_tokens_per_request'] if output is None else output,
        max_context_tokens_per_request=reference['max_context_tokens_per_request'] if context is None else context,
        prefill_chunk_tokens=(reference['prefill_chunk_tokens'] or len(ids)) if prefill_chunk is None else prefill_chunk)
    limits = dict(batch=(1,512), output_tokens_per_request=(16,2048),
        max_context_tokens_per_request=(1,4096), prefill_chunk_tokens=(1,4096))
    for key, (low,high) in limits.items():
        if type(selected[key]) is not int or not low <= selected[key] <= high:
            raise ValueError(f'Invalid candidate profile {key}')
    if len(ids)+selected['output_tokens_per_request'] > selected['max_context_tokens_per_request']:
        raise ValueError('Candidate context shortens the original input or requested output')
    return dict(schema=1, scope='Unvalidated shape for sequential engine statistics, not a completed workload',
        reference_sha256=hashlib.sha256(reference_bytes).hexdigest(),
        input_ids_sha256=hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest(),
        input_tokens_per_request=len(ids), **selected)


def validate_candidate_shape(specification, reference, reference_bytes):
    if not isinstance(specification,dict) or specification.get('schema') != 1:
        raise ValueError('Unsupported candidate engine-profile specification')
    expected = candidate_shape(reference,reference_bytes,
        batch=specification.get('batch'), output=specification.get('output_tokens_per_request'),
        context=specification.get('max_context_tokens_per_request'),
        prefill_chunk=specification.get('prefill_chunk_tokens'))
    if specification != expected:
        raise ValueError('Candidate shape/reference/input identity mismatch')
    return expected
