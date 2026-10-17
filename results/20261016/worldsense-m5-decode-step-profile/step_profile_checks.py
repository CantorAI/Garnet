"""Integrity checks, not timing performance gates. Clock tolerance is0.1ms."""
import math

def validate(call,stops=None):
 t=call['timings'];trace=t['decode_step_profile'];assert trace['version']==1 and 'decode_step_profile'in call['backend']
 rows=trace['steps'];tokens=call['generated_token_ids'];assert len(tokens)==call['generated_tokens']
 assert max(0,len(tokens)-1)<=len(rows)<=len(tokens)
 for index,row in enumerate(rows):
  assert row['index']==index and row['input_token']==tokens[index]
  assert row['position']==rows[0]['position']+index
  if not call.get('image_preprocess_backend'):assert row['position']==t['input_tokens']+index
  if index+1<len(tokens):assert row['output_prediction']==tokens[index+1]
  elif stops is not None:assert row['output_prediction']in stops
  for key in ['host_step_seconds','native_inference_ms','prepare_ms','encode_ms','gpu_ms','wait_ms']:assert math.isfinite(row[key])and row[key]>=0
  assert row['host_step_seconds']*1000+.1>=row['native_inference_ms']
  assert row['prepare_ms']+row['encode_ms']+row['wait_ms']<=row['native_inference_ms']+.1
 for key in ['prepare_ms','encode_ms','gpu_ms','wait_ms']:assert abs(sum(r[key]for r in rows)-t['decode_profile'][key])<1e-6
 assert sum(r['host_step_seconds']for r in rows)<=t['decode_seconds']+.0001
 return len(rows)
