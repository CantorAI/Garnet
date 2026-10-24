import hashlib,json,tempfile,unittest,threading,time
from http.server import BaseHTTPRequestHandler,HTTPServer
from pathlib import Path
from benchmark_stream import TokenEvents,payload,request
from probe_reference_http import image_png

class StreamingAccountingTests(unittest.TestCase):
    def feed(self,state,n,now,text='',finish=None):
        state.feed(json.dumps({'choices':[{'index':0,'delta':{'content':text},'finish_reason':finish}],
            'usage':{'prompt_tokens':100,'completion_tokens':n}}),now)
    def test_server_counts_not_chunks_and_batched_arrivals_retained(self):
        state=TokenEvents();self.feed(state,0,.5);self.feed(state,1,1.,'a');self.feed(state,3,2.,'bc','length');state.feed('[DONE]',2.1)
        result=state.result(0.,2.2,3)
        self.assertEqual(result['generated_tokens'],3);self.assertEqual(result['largest_arrival_batch_tokens'],2)
        self.assertFalse(result['single_token_arrivals']);self.assertEqual(result['last_token'],2.)
    def test_final_usage_only_does_not_fabricate_decode_interval(self):
        state=TokenEvents();self.feed(state,3,2.,'abc','length');state.feed('[DONE]',2.1)
        with self.assertRaises(ValueError):state.result(0.,2.2,3)
    def test_rejects_backward_counts_incomplete_and_cached_prompt(self):
        state=TokenEvents();self.feed(state,2,1.)
        with self.assertRaises(ValueError):self.feed(state,1,2.)
        with self.assertRaises(ValueError):state.result(0.,2.,3)
        with self.assertRaises(ValueError):state.feed(json.dumps({'usage':{'prompt_tokens':100,'completion_tokens':2,'prompt_tokens_details':{'cached_tokens':1}}}),2.)
    def test_real_http_framing_and_failed_body_retention(self):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*args):pass
            def do_POST(self):
                self.rfile.read(int(self.headers['Content-Length']))
                if self.path.startswith('/bad/'):
                    self.send_response(503);self.end_headers();self.wfile.write(b'FIXTURE_UNAVAILABLE');return
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
                for n in (1,2):
                    obj={'choices':[{'index':0,'delta':{'content':'a'},'finish_reason':'length' if n==2 else None}],
                        'usage':{'prompt_tokens':10,'completion_tokens':n}}
                    self.wfile.write(('data: '+json.dumps(obj)+'\n\n').encode());self.wfile.flush();time.sleep(.005)
                self.wfile.write(b'data: [DONE]\n\n');self.wfile.flush()
        server=HTTPServer(('127.0.0.1',0),Handler);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);url='http://127.0.0.1:'+str(server.server_port)
                result=request(url,{},root,'synthetic-http',2)
                self.assertEqual(result['status'],'completed');self.assertEqual(result['generated_tokens'],2)
                self.assertLess(result['first_token'],result['last_token'])
                failed=request(url+'/bad',{},root,'synthetic-failure',2)
                self.assertEqual(failed['status'],'failed')
                self.assertIn('FIXTURE_UNAVAILABLE',(root/'synthetic-failure-events.jsonl').read_text())
        finally:server.shutdown();server.server_close();thread.join()

    def test_nonce_image_fixture_is_stable_and_uncached(self):
        self.assertEqual(hashlib.sha256(image_png()).hexdigest(),'ad17ed2fc2311d796ef00f78cad2545ac9b9a3626c4a60e1c1891931933665e5')
        first,image=payload('image','image-c1-r0',10001,'model',128)
        again,repeated=payload('image','image-c1-r0',10001,'model',128)
        self.assertEqual(first,again);self.assertEqual(image,repeated);self.assertNotEqual(image,image_png(10002))

    def test_jpeg_profile_preserves_wire_bytes_order_and_nonce(self):
        import base64
        from io import BytesIO
        from PIL import Image
        body,image=payload('image','s3-image-c1-r0',321001,'model',128,'jpeg','image-first')
        content=body['messages'][0]['content']
        self.assertEqual(content[0]['type'],'image_url');self.assertEqual(content[1]['type'],'text')
        url=content[0]['image_url']['url'];self.assertTrue(url.startswith('data:image/jpeg;base64,'))
        self.assertEqual(base64.b64decode(url.split(',',1)[1]),image)
        with Image.open(BytesIO(image)) as decoded:self.assertEqual(decoded.size,(224,224));self.assertEqual(decoded.format,'JPEG')
        self.assertEqual(image,payload('image','s3-image-c1-r0',321001,'model',128,'jpeg','image-first')[1])
        self.assertNotEqual(image,payload('image','s3-image-c1-r1',321002,'model',128,'jpeg','image-first')[1])
        self.assertEqual(payload('text','key',1,'model',128),payload('text','key',1,'model',128,'jpeg','image-first'))
        with self.assertRaises(ValueError):payload('image','key',1,'model',128,'gif')
if __name__=='__main__':unittest.main()
