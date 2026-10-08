"""Independent shape/provenance/rejection boundaries; no GPU execution."""
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'tools/gpt_oss'))
from engine_profile_shape import candidate_shape, validate_candidate_shape

reference = dict(batch=256,input_token_ids=list(range(256)),input_tokens_per_request=256,
    output_tokens_per_request=512,max_context_tokens_per_request=1024,prefill_chunk_tokens=16)
raw = json.dumps(reference).encode()
shape = candidate_shape(reference,raw,batch=448,context=768,prefill_chunk=8)
assert (shape['batch'],shape['input_tokens_per_request'],shape['output_tokens_per_request'],
    shape['max_context_tokens_per_request'],shape['prefill_chunk_tokens']) == (448,256,512,768,8)
assert shape['reference_sha256'] == hashlib.sha256(raw).hexdigest()
assert validate_candidate_shape(shape,reference,raw) == shape
for overrides in ({'batch':513},{'batch':True},{'batch':0},{'output':15},
                  {'output':2049},{'context':767},{'context':4097},{'prefill_chunk':0}):
    try:
        candidate_shape(reference,raw,**overrides)
    except ValueError:
        pass
    else:
        raise AssertionError(overrides)
for key,value in (('reference_sha256','other'),('input_ids_sha256','other'),
                  ('input_tokens_per_request',255),('schema',2),('scope','completed')):
    corrupt = dict(shape,**{key:value})
    try:
        validate_candidate_shape(corrupt,reference,raw)
    except ValueError:
        pass
    else:
        raise AssertionError(key)
for changed in (dict(reference,input_tokens_per_request=255),
                dict(reference,input_token_ids=[]),dict(reference,input_token_ids=[-1]*256)):
    try:
        candidate_shape(changed,json.dumps(changed).encode(),batch=384)
    except ValueError:
        pass
    else:
        raise AssertionError('Invalid original input identity admitted')
long = dict(reference,input_token_ids=[1]*2005,input_tokens_per_request=2005,
    batch=128,max_context_tokens_per_request=2560,prefill_chunk_tokens=32)
long_raw = json.dumps(long).encode()
long_shape = candidate_shape(long,long_raw,batch=144,prefill_chunk=16)
assert long_shape['input_tokens_per_request'] == 2005
assert long_shape['max_context_tokens_per_request'] == 2560
for source,payload,batch,context,chunk,count,tail,pad in (
        (reference,raw,448,768,9,29,4,5),
        (long,long_raw,144,2560,28,72,17,11)):
    candidate = candidate_shape(source,payload,batch=batch,context=context,prefill_chunk=chunk)
    assert validate_candidate_shape(candidate,source,payload) == candidate
    length = candidate['input_tokens_per_request']
    assert candidate['input_ids_sha256'] == hashlib.sha256(
        json.dumps(source['input_token_ids'],separators=(',',':')).encode()).hexdigest()
    assert (batch*chunk,(length+chunk-1)//chunk,length%chunk,(-length)%chunk)==(4032,count,tail,pad)
    assert candidate['output_tokens_per_request']==512 and length+512<=context
    malformed=dict(candidate,input_tokens_per_request=length-pad)
    try:validate_candidate_shape(malformed,source,payload)
    except ValueError:pass
    else:raise AssertionError('Irregular candidate silently shortened actual input')
print('Candidate shapes preserve exact input/reference identity; malformed/shortened shapes reject PASS')
