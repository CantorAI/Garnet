"""Matched JPEG reference/token oracles; smoke coverage, never a timing benchmark."""
import argparse,base64,hashlib,json
from pathlib import Path
from urllib.request import Request,urlopen
from PIL import Image,ImageDraw

def main():
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18081');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
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
        payload={'model':'qwen38-27b-nvfp4','messages':[{'role':'user','content':content}],
            'max_tokens':32,'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':False},
            'return_prompt_token_ids':True,'return_token_ids':True}
        wire=json.dumps(payload).encode();(a.output/(name+'-request.json')).write_bytes(wire)
        with urlopen(Request(a.url+'/v1/chat/completions',data=wire,headers={'Content-Type':'application/json'}),timeout=180) as response:raw=response.read()
        (a.output/(name+'-response.json')).write_bytes(raw);result=json.loads(raw);choice=result['choices'][0]
        ids=choice['prompt_token_ids'];generated=choice['response_token_ids'];text=choice['message']['content'].strip()
        if not ids or not generated or text.lower().rstrip('.')!=want:raise ValueError('Reference image smoke failed: '+name)
        if ids.count(248056)!=64:raise ValueError('Reference must use64 image tokens')
        reports.append({'name':name,'image_file':path.name,'question':question,'expected_color':want,
            'prompt_ids':ids,'reference_generated_ids':generated,'reference_text':text,
            'image_sha256':hashlib.sha256(data).hexdigest(),'request_sha256':hashlib.sha256(wire).hexdigest(),
            'response_sha256':hashlib.sha256(raw).hexdigest(),'usage':result['usage']})
        print('MATCHED_REFERENCE_IMAGE_PASS',name,text,flush=True)
    oracle={'cases':reports,'grid':[1,16,16],'context_capacity':4096,
        'scope':'Four matched JPEG color/spatial smoke cases only; not broad VLM quality or performance'}
    (a.output/'oracle.json').write_text(json.dumps(oracle,indent=2))

if __name__=='__main__':main()
