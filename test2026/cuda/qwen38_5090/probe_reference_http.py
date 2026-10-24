"""Actual text/image reference smoke requests, not a performance benchmark."""
import argparse,base64,hashlib,json,struct,zlib
from pathlib import Path
from urllib.request import Request,urlopen


def image_png(seed=0,nonce_pattern='pixel'):
    if nonce_pattern not in ('pixel','binary-grid') or type(seed) is not int or not 0<=seed<2**24:raise ValueError('Invalid image nonce')
    def chunk(kind,data):
        return struct.pack('>I',len(data))+kind+data+struct.pack('>I',zlib.crc32(kind+data)&0xffffffff)
    def pixel(x,y):
        if nonce_pattern=='binary-grid' and x>=160 and y>=200:
            bit=((y-200)//8)*8+(x-160)//8
            return (255,255,255) if seed&(1<<bit) else (0,0,0)
        if nonce_pattern=='pixel' and seed and x==223 and y==223:return (seed&255,(seed>>8)&255,(seed>>16)&255)
        return (0,0,255) if x<56 and y<56 else (255,0,0)
    rows=b''.join(b'\0'+b''.join(bytes(pixel(x,y)) for x in range(224)) for y in range(224))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',224,224,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(rows))+chunk(b'IEND',b'')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='http://127.0.0.1:18080');p.add_argument('--model',default='qwen38-27b-nvfp4');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    with urlopen(a.url+'/health',timeout=10) as r: print('HEALTH',r.status)
    png=image_png();(a.output/'red-blue-224.png').write_bytes(png)
    cases=[('text','What is 2 + 2? Answer with the number only.'),
        ('image',[{'type':'text','text':'What is the dominant color in this image? Answer with one color word.'},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,'+base64.b64encode(png).decode()}}])]
    for name,content in cases:
        payload={'model':a.model,'messages':[{'role':'user','content':content}],
            'max_tokens':64,'temperature':0,'stream':False,'chat_template_kwargs':{'enable_thinking':False}}
        raw=json.dumps(payload).encode();(a.output/(name+'-request.json')).write_bytes(raw)
        request=Request(a.url+'/v1/chat/completions',data=raw,headers={'Content-Type':'application/json'})
        with urlopen(request,timeout=180) as r: response=r.read()
        (a.output/(name+'-response.json')).write_bytes(response)
        result=json.loads(response)
        print(json.dumps({'case':name,'request_sha256':hashlib.sha256(raw).hexdigest(),
            'image_sha256':hashlib.sha256(png).hexdigest() if name=='image' else None,
            'choices':result.get('choices'),'usage':result.get('usage'),
            'scope':'Actual reference smoke response; no timing or quality benchmark'}))
