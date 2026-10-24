"""Preserve the original native VLM/prompt gate for every research decoder mode."""
import json
from urllib.request import Request,urlopen

def validate_saved_vlm(root,out,url):
 native=json.loads((root/'results/native-full-vlm-fused16-checks.json').read_text());expected={c['name']:c for c in native['cases']}
 oracle=json.loads((root/'results/matched-vlm-reference/oracle.json').read_text());rows=[]
 for case in oracle['cases']:
  name=case['name'];raw=(root/'results/matched-vlm-reference'/(name+'-request.json')).read_bytes();(out/(name+'-request.json')).write_bytes(raw)
  with urlopen(Request(url+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'}),timeout=300) as response:result=response.read()
  (out/(name+'-response.json')).write_bytes(result);choice=json.loads(result)['choices'][0]
  row={'case':name,'prompt_ids_match':choice['prompt_token_ids']==case['prompt_ids'],'native_regression_match':choice['response_token_ids']==expected[name]['generated_ids'],
   'reference_exact_parity':choice['response_token_ids']==case['reference_generated_ids'],'semantic_pass':choice['message']['content'].strip().lower().rstrip('.')==case['expected_color']}
  rows.append(row);(out/'correctness.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)
  if not row['prompt_ids_match'] or not row['native_regression_match'] or not row['semantic_pass']:raise ValueError('Native HTTP transport changed retained native VLM behavior')
 return rows
