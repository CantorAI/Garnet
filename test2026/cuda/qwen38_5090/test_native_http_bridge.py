"""CPU protocol fixtures only; these are never model performance measurements."""
import json,sys,tempfile,threading,time,unittest
from pathlib import Path
from urllib.request import Request,urlopen
from native_http_bridge import Worker,Handler,ThreadingHTTPServer,validate_body
from benchmark_stream import request,payload

FIXTURE='''import sys,json,time
print('fixture startup log',flush=True)
print(json.dumps({'event':'ready'}),flush=True)
for line in sys.stdin:
 r=json.loads(line);k=r['id']
 for n,text in [(1,'a\\ufffd'),(2,'aé')]:
  print(json.dumps({'event':'token','id':k,'token_id':n,'sampled_tokens':n,'prompt_tokens':3,'text':text,'stop':False}),flush=True)
  time.sleep(.03)
 print(json.dumps({'event':'completed','id':k,'generated_ids':[1,2],'prompt_ids':[3,4,5],'text':'aé','finish_reason':'length'}),flush=True)
'''
class BridgeTests(unittest.TestCase):
 def test_incremental_real_pipe_and_http_accounting_unicode_and_final_ids(self):
  with tempfile.TemporaryDirectory() as tmp:
   root=Path(tmp);script=root/'fixture.py';script.write_text(FIXTURE,encoding='utf-8')
   worker=Worker([sys.executable,'-u',str(script)],root)
   server=ThreadingHTTPServer(('127.0.0.1',0),Handler);server.worker=worker
   thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
   try:
    deadline=time.monotonic()+10
    while not worker.healthy() and time.monotonic()<deadline:time.sleep(.01)
    self.assertTrue(worker.healthy());url='http://127.0.0.1:'+str(server.server_port)
    body,_=payload('text','fixture',1,'qwen38-27b-nvfp4',2)
    result=request(url,body,root,'synthetic-bridge',2)
    self.assertEqual(result['status'],'completed');self.assertEqual(result['generated_tokens'],2)
    self.assertEqual(result['text'],'aé');self.assertTrue(result['single_token_arrivals'])
    self.assertGreater(result['last_token']-result['first_token'],.015)
    body['stream']=False
    with urlopen(Request(url+'/v1/chat/completions',data=json.dumps(body).encode(),headers={'Content-Type':'application/json'})) as response:final=json.load(response)
    self.assertEqual(final['choices'][0]['message']['content'],'aé');self.assertEqual(final['choices'][0]['response_token_ids'],[1,2])
    self.assertIn('fixture startup log',(root/'worker-stdout.jsonl').read_text())
   finally:server.shutdown();server.server_close();thread.join();worker.close()
 def test_profile_rejects_wrong_model_thinking_and_image_order(self):
  body,_=payload('text','fixture',1,'qwen38-27b-nvfp4',2);validate_body(body)
  body['chat_template_kwargs']['enable_thinking']=True
  with self.assertRaises(ValueError):validate_body(body)
  image,_=payload('image','fixture',1,'qwen38-27b-nvfp4',2)
  with self.assertRaises(ValueError):validate_body(image)
  image,_=payload('image','fixture',1,'qwen38-27b-nvfp4',2,'jpeg','image-first');validate_body(image)
if __name__=='__main__':unittest.main()
