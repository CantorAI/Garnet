"""Synthetic CPU HTTP/counter checks, not model or cache performance results."""
import json,tempfile,threading,unittest
from pathlib import Path
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from vllm_cache_control import parse_metrics,check_url,reset_idle_caches,verify_after_run

def metrics(running=0,waiting=0,queries=5,hits=2):
 return ('# fixture only\n'+f'vllm:num_requests_running{{engine="0"}} {running}\n'+f'vllm:num_requests_waiting{{engine="0"}} {waiting}\n'+f'vllm:mm_cache_queries_total{{engine="0"}} {queries}\n'+f'vllm:mm_cache_hits_total{{engine="0"}} {hits}\nprocess_start_time_seconds 123\n').encode()

class CacheControlTests(unittest.TestCase):
 def test_counter_names_labels_and_required_fields(self):
  self.assertEqual(parse_metrics(metrics())['hits'],2)
  for wire in (metrics().replace(b'process_start_time_seconds',b'other_start'),metrics().replace(b' 123',b' NaN'),metrics()+b'vllm:mm_cache_hits 2\n'):
   with self.assertRaises(ValueError):parse_metrics(wire)
 def test_literal_loopback_only(self):
  for url in ('https://127.0.0.1','http://localhost','http://127.0.0.1/path','http://u:p@127.0.0.1','http://127.0.0.1?q=x'):
   with self.assertRaises(ValueError):check_url(url)
 def test_busy_server_never_receives_reset(self):
  from unittest.mock import MagicMock
  response=MagicMock();response.__enter__.return_value.read.return_value=b'{"data":[{"id":"qwen38-27b-int4"}]}'
  with tempfile.TemporaryDirectory() as tmp,patch('vllm_cache_control.urlopen',return_value=response) as opening,patch('vllm_cache_control.snapshot',return_value=parse_metrics(metrics(running=1))),patch('vllm_cache_control.time.monotonic',side_effect=[0,2]):
   with self.assertRaises(TimeoutError):reset_idle_caches('http://127.0.0.1:18080',Path(tmp)/'control',timeout=1)
   self.assertEqual(opening.call_count,1) # model GET only
 def test_both_reset_endpoints_and_zero_new_hits(self):
  state={'queries':5,'hits':2,'calls':[]}
  class Fixture(BaseHTTPRequestHandler):
   def log_message(self,*args):pass
   def do_GET(self):
    wire=b'{"data":[{"id":"qwen38-27b-int4"}]}' if self.path=='/v1/models' else metrics(queries=state['queries'],hits=state['hits'])
    self.send_response(200);self.send_header('Content-Length',str(len(wire)));self.end_headers();self.wfile.write(wire)
   def do_POST(self):
    state['calls'].append(self.path)
    if self.path=='/reset_mm_cache':state['queries']=state['hits']=0
    self.send_response(200);self.send_header('Content-Length','0');self.end_headers()
  server=ThreadingHTTPServer(('127.0.0.1',0),Fixture);thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
  try:
   with tempfile.TemporaryDirectory() as tmp:
    out=Path(tmp)/'control';url='http://127.0.0.1:'+str(server.server_port);report=reset_idle_caches(url,out)
    self.assertEqual(state['calls'],['/reset_mm_cache','/reset_encoder_cache']);self.assertEqual(report['status'],'reset_acknowledged')
    state['queries']=64;result=verify_after_run(url,out,report['after_reset']);self.assertEqual(result['query_delta'],64);self.assertEqual(result['hit_delta'],0)
    with self.assertRaises(FileExistsError):reset_idle_caches(url,out)
  finally:server.shutdown();server.server_close();thread.join()
 def test_new_hits_fail_even_with_successful_reset_acknowledgement(self):
  with tempfile.TemporaryDirectory() as tmp,patch('vllm_cache_control.snapshot',return_value=parse_metrics(metrics(queries=64,hits=1))):
   with self.assertRaises(ValueError):verify_after_run('http://127.0.0.1:18080',Path(tmp),parse_metrics(metrics(queries=0,hits=0)))

if __name__=='__main__':unittest.main()
