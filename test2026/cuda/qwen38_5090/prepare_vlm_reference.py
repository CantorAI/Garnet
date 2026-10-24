"""Matched JPEG reference/token oracles; smoke coverage, never a timing benchmark."""
import argparse,base64,hashlib,json
from pathlib import Path
from urllib.request import Request,urlopen
from PIL import Image,ImageDraw

def reference_tokens(result,engine):
    """Installed primary API layouts differ; never infer IDs from text/chunks."""
    if engine not in ('sglang','vllm'):raise ValueError('Unsupported reference engine')
    if len(result.get('choices',[]))!=1:raise ValueError('One reference choice required')
    choice=result['choices'][0]
    if engine=='sglang':ids=choice.get('prompt_token_ids');generated=choice.get('response_token_ids')
    else:ids=result.get('prompt_token_ids');generated=choice.get('token_ids')
    for values in (ids,generated):
        if not isinstance(values,list) or not values or any(type(v) is not int or not 0<=v<248320 for v in values):raise ValueError('Missing/invalid actual reference token IDs')
    if len(ids)+len(generated)>4096:raise ValueError('Reference context capacity exceeded')
    return ids,generated

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18081');p.add_argument('--output',type=Path,required=True);p.add_argument('--engine',choices=['sglang','vllm'],default='sglang');p.add_argument('--model',choices=['qwen38-27b-nvfp4','qwen38-27b-int4','qwen35-4b-bf16'],default='qwen38-27b-nvfp4');a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=True)
    prompt='What is the dominant color in this image? Answer with one color word.'
    cases=[]
    for name,rgb in [('red',(255,0,0)),('green',(0,255,0)),('blue',(0,0,255))]:
        cases.append((name,Image.new('RGB',(224,224),rgb),prompt,name))
    layout=Image.new('RGB',(224,224),(255,0,0));ImageDraw.Draw(layout).rectangle((0,112,223,223),fill=(0,0,255))
    cases.append(('bottom-blue',layout,'What color is the bottom half of this image? Answer with one color word.','blue'))
    reports=[]
    for name,image,question,want in cases:
        path=a.output/(name+'.jpg');image.save(path,quality=100,subsampling=0);data=path.read_bytes()
        content=[{'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+base64.b64encode(data).decode()}},
                 {'type':'text','text':question}]
        payload={'model':a.model,'messages':[{'role':'user','content':content}],
            'max_tokens':32,'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':False},
            'return_prompt_token_ids':True,'return_token_ids':True}
        if a.engine=='vllm':del payload['return_prompt_token_ids']
        wire=json.dumps(payload).encode();(a.output/(name+'-request.json')).write_bytes(wire)
        with urlopen(Request(a.url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=180) as response:raw=response.read()
        (a.output/(name+'-response.json')).write_bytes(raw);result=json.loads(raw);choice=result['choices'][0]
        ids,generated=reference_tokens(result,a.engine);text=choice['message']['content'].strip()
        if not ids or not generated or text.lower().rstrip('.')!=want:raise ValueError('Reference image smoke failed: '+name)
        if ids.count(248056)!=64:raise ValueError('Reference must use64 image tokens')
        reports.append({'name':name,'image_file':path.name,'question':question,'expected_color':want,
            'prompt_ids':ids,'reference_generated_ids':generated,'reference_text':text,
            'image_sha256':hashlib.sha256(data).hexdigest(),'request_sha256':hashlib.sha256(wire).hexdigest(),
            'response_sha256':hashlib.sha256(raw).hexdigest(),'usage':result['usage']})
        print('MATCHED_REFERENCE_IMAGE_PASS',name,text,flush=True)
    oracle={'cases':reports,'model':a.model,'engine':a.engine,'grid':[1,16,16],'context_capacity':4096,
        'scope':'Four matched JPEG color/spatial smoke cases only; not broad VLM quality or performance'}
    (a.output/'oracle.json').write_text(json.dumps(oracle,indent=2))

if __name__=='__main__':main()
