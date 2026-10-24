import copy, json, subprocess, sys, tempfile, threading, unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from reference_mtp_equivalence import compare_pair


class ReferenceMTPGates(unittest.TestCase):
    def test_actual_capture_compare_and_failure_artifacts(self):
        class Handler(BaseHTTPRequestHandler):
            mismatch=False
            def log_message(self,*args):pass
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if body['stream'] or 'stream_options' in body or not body.get('return_token_ids'):
                    self.send_error(400);return
                image=isinstance(body['messages'][0]['content'],list)
                result={'prompt_token_ids':[1]*(104 if image else 41),
                        'choices':[{'token_ids':[4,6 if Handler.mismatch else 5]}],
                        'usage':{'completion_tokens':2}}
                wire=json.dumps(result).encode();self.send_response(200)
                self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
        try:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);script=Path(__file__).with_name('reference_mtp_equivalence.py')
                base=[sys.executable,str(script),'--engine','vllm','--url','http://127.0.0.1:'+str(server.server_port)]
                ordinary=root/'ordinary';trial=root/'trial';failed=root/'failed'
                a=subprocess.run(base+['--output',str(ordinary)],capture_output=True,text=True,timeout=30)
                self.assertEqual(a.returncode,0,a.stderr)
                oracle=ordinary/'equivalence.json'
                b=subprocess.run(base+['--output',str(trial),'--ordinary-oracle',str(oracle)],capture_output=True,text=True,timeout=30)
                self.assertEqual(b.returncode,0,b.stderr)
                report=json.loads((trial/'equivalence.json').read_text())
                self.assertEqual(report['status'],'passed');self.assertEqual(len(report['cases']),16)
                self.assertTrue(all(r['exact_match'] for r in report['cases']))
                Handler.mismatch=True
                c=subprocess.run(base+['--output',str(failed),'--ordinary-oracle',str(oracle)],capture_output=True,text=True,timeout=30)
                self.assertNotEqual(c.returncode,0)
                report=json.loads((failed/'equivalence.json').read_text())
                self.assertEqual(report['status'],'failed');self.assertFalse(report['cases'][0]['exact_match'])
                self.assertTrue((failed/'s3-text-c1-r0-response.json').exists())
                self.assertEqual(len(list(ordinary.glob('*-request.json'))),16)
        finally:server.shutdown();server.server_close();thread.join(timeout=5)

    def test_full_ids_required(self):
        a={'engine':'vllm','key':'text','semantic_request_sha256':'same','prompt_ids':[1,2],'generated_ids':[3,4,5]}
        self.assertTrue(compare_pair(a,copy.deepcopy(a)))
        for field,value in [('prompt_ids',[1,9]),('generated_ids',[3,4,6]),('generated_ids',[])]:
            b=copy.deepcopy(a);b[field]=value;self.assertFalse(compare_pair(a,b))

    def test_engine_and_request_identity_required(self):
        a={'engine':'vllm','key':'text','semantic_request_sha256':'same','prompt_ids':[1],'generated_ids':[2]}
        for field,value in [('engine','sglang'),('key','image'),('semantic_request_sha256','different')]:
            b=copy.deepcopy(a);b[field]=value
            with self.assertRaises(ValueError):compare_pair(a,b)


if __name__=='__main__':unittest.main()
