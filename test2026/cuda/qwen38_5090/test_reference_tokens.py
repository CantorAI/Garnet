"""Reference API layout/invalid-ID fixtures, not actual model outputs."""
import unittest
from prepare_vlm_reference import reference_tokens
class ReferenceTokenLayoutTests(unittest.TestCase):
 def test_installed_engine_fields_are_distinct_and_explicit(self):
  sg={'choices':[{'prompt_token_ids':[1,248056],'response_token_ids':[5844,248046]}]}
  vl={'prompt_token_ids':[1,248056],'choices':[{'token_ids':[5844,248046]}]}
  self.assertEqual(reference_tokens(sg,'sglang'),reference_tokens(vl,'vllm'))
  with self.assertRaises(ValueError):reference_tokens(sg,'vllm')
  with self.assertRaises(ValueError):reference_tokens(vl,'sglang')
 def test_no_missing_ids_boolean_or_text_inference(self):
  for values in (None,[],[True],[-1],[248320]):
   with self.assertRaises(ValueError):reference_tokens({'prompt_token_ids':[1],'choices':[{'token_ids':values,'message':{'content':'Red'}}]},'vllm')
  with self.assertRaises(ValueError):reference_tokens({'prompt_token_ids':[1]*4096,'choices':[{'token_ids':[1]}]},'vllm')
class ReferenceModelRoutingTests(unittest.TestCase):
 def test_selected_checkpoint_name_reaches_every_reference_request(self):
  # Synthetic HTTP protocol only; this does not load or qualify a model.
  import json,subprocess,sys,tempfile,threading
  from pathlib import Path
  from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
  for model in ('qwen38-27b-int4','qwen35-4b-bf16'):
   seen=[];colors=['red','green','blue','blue']
   class Fixture(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def do_POST(self):
     body=json.loads(self.rfile.read(int(self.headers['Content-Length'])));seen.append(body)
     result={'prompt_token_ids':[1]+[248056]*64,'choices':[{'token_ids':[99],'message':{'content':colors[len(seen)-1]}}],'usage':{'prompt_tokens':65,'completion_tokens':1}}
     wire=json.dumps(result).encode();self.send_response(200);self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
   server=ThreadingHTTPServer(('127.0.0.1',0),Fixture);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
   try:
    with tempfile.TemporaryDirectory() as tmp:
     subprocess.run([sys.executable,str(Path(__file__).with_name('prepare_vlm_reference.py')),'--engine','vllm','--model',model,'--url','http://127.0.0.1:'+str(server.server_port),'--output',tmp],check=True,capture_output=True)
     self.assertEqual(len(seen),4);self.assertTrue(all(body['model']==model for body in seen))
     self.assertEqual(json.loads((Path(tmp)/'oracle.json').read_text())['model'],model)
   finally:server.shutdown();server.server_close();thread.join()
if __name__=='__main__':unittest.main()
