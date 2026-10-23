"""Save SGLang's actual chat token IDs for independent native decoder smoke."""
import argparse,hashlib,json
from pathlib import Path
from urllib.request import Request,urlopen
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18081');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    payload={'model':'qwen38-27b-nvfp4','messages':[{'role':'user','content':'What is 2 + 2? Answer with the number only.'}],
        'max_tokens':64,'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':False},
        'return_prompt_token_ids':True,'return_token_ids':True}
    wire=json.dumps(payload).encode();(a.output/'text-token-request.json').write_bytes(wire)
    with urlopen(Request(a.url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=180) as response:raw=response.read()
    (a.output/'text-token-response.json').write_bytes(raw);result=json.loads(raw);choice=result['choices'][0]
    ids=choice['prompt_token_ids'];generated=choice['response_token_ids']
    if not ids or not generated or choice['message']['content'].strip()!='4':raise ValueError('Reference token smoke failed')
    output={'prompt_ids':ids,'reference_generated_ids':generated,'reference_text':choice['message']['content'],
        'request_sha256':hashlib.sha256(wire).hexdigest(),'response_sha256':hashlib.sha256(raw).hexdigest(),
        'scope':'Reference chat tokenization and basic arithmetic smoke only; not broad model quality or timing'}
    (a.output/'text-token-oracle.json').write_text(json.dumps(output,indent=2));print(json.dumps(output))
