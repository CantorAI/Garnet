"""Loopback research bridge to actual native token events; single slot, queued clients.

General Garnet serving admission is separate. Does not synthesize token timings.
"""
import argparse,json,queue,subprocess,threading,time,uuid
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
from pathlib import Path

class Worker:
    def __init__(self,command,root):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True)
        self.events=queue.Queue();self.lock=threading.Lock();self.ready=False;self.failure=None
        self.stderr=(self.root/'worker-stderr.log').open('ab',buffering=0)
        self.process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.stderr,bufsize=0)
        self.reader=threading.Thread(target=self._read,daemon=True);self.reader.start()
    def _read(self):
        try:
            with (self.root/'worker-stdout.jsonl').open('a',encoding='utf-8') as raw:
                for line in self.process.stdout:
                    now=time.monotonic();text=line.decode('utf-8',errors='replace').rstrip('\r\n')
                    raw.write(json.dumps({'received':now,'line':text})+'\n');raw.flush()
                    try:event=json.loads(text)
                    except ValueError:continue
                    if not isinstance(event,dict):continue
                    kind=event.get('event')
                    if kind=='ready':self.ready=True;self.ready_info=event
                    elif kind in ('token','completed','error'):self.events.put(event)
        except Exception as error:self.failure=repr(error)
        finally:
            self.ready=False;self.events.put({'event':'worker_exit','error':self.failure})
    def healthy(self):return self.ready and self.process.poll() is None
    def generate(self,body,key):
        # Queue time is intentionally included in client-observed TTFT.
        with self.lock:
            if not self.healthy():raise RuntimeError('Native worker is not ready')
            wire=(json.dumps({'id':key,'body':body},separators=(',',':'))+'\n').encode()
            while wire:
                written=self.process.stdin.write(wire)
                if not written:raise OSError('Worker request pipe closed')
                wire=wire[written:]
            self.process.stdin.flush()
            count=0
            while True:
                try:event=self.events.get(timeout=300)
                except queue.Empty:
                    self.ready=False;raise TimeoutError('Native worker event deadline')
                if event['event']=='worker_exit':raise RuntimeError('Native worker exited: '+str(event.get('error')))
                if event.get('id')!=key:self.ready=False;raise ValueError('Worker request ID mismatch')
                if event['event']=='error':self.ready=False;raise RuntimeError(event['error'])
                if event['event']=='token':
                    count+=1
                    if event['sampled_tokens']!=count:raise ValueError('Worker token count mismatch')
                elif len(event['generated_ids'])!=count:raise ValueError('Final token count mismatch')
                yield event
                if event['event']=='completed':return
    def close(self):
        self.process.stdin.close()
        self.process.wait(timeout=120);self.reader.join(timeout=10);self.process.stdout.close();self.stderr.close()

def validate_body(body,model='qwen38-27b-nvfp4'):
    if not isinstance(body,dict) or body.get('model')!=model or body.get('temperature')!=0:
        raise ValueError('Study requires the pinned model and greedy generation')
    if body.get('chat_template_kwargs')!={'enable_thinking':False}:raise ValueError('Thinking must be disabled')
    limit=body.get('max_tokens')
    if type(limit) is not int or not 2<=limit<=1024:raise ValueError('Output limit outside study profile')
    messages=body.get('messages')
    if not isinstance(messages,list) or len(messages)!=1 or messages[0].get('role')!='user':raise ValueError('One user message required')
    content=messages[0].get('content')
    if isinstance(content,str):return
    if not isinstance(content,list) or len(content)!=2 or content[0].get('type')!='image_url' or content[1].get('type')!='text':
        raise ValueError('Image profile requires JPEG first, then text')
    if not isinstance(content[1].get('text'),str) or not content[0].get('image_url',{}).get('url','').startswith('data:image/jpeg;base64,'):
        raise ValueError('Inline JPEG required')

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):pass
    def reply(self,status,body):
        raw=json.dumps(body).encode();self.send_response(status);self.send_header('Content-Type','application/json');self.send_header('Content-Length',str(len(raw)));self.end_headers();self.wfile.write(raw)
    def do_GET(self):
        if self.path!='/health':self.reply(404,{'error':'Unknown study endpoint'});return
        healthy=self.server.worker.healthy()
        info={'research_only':True,'batch':1,'context_capacity':4096,'queued_concurrency':True}
        if healthy:info['native_worker']=self.server.worker.ready_info
        self.reply(200 if healthy else 503,info)
    def do_POST(self):
        if self.path!='/v1/chat/completions':self.reply(404,{'error':'Unknown study endpoint'});return
        try:
            size=int(self.headers.get('Content-Length','0'))
            if not 1<=size<=2*1024*1024:raise ValueError('Request size outside study bounds')
            body=json.loads(self.rfile.read(size));validate_body(body,getattr(self.server,'model','qwen38-27b-nvfp4'))
            if not self.server.worker.healthy():self.reply(503,{'error':'Native worker not ready'});return
        except Exception as error:self.reply(400,{'error':str(error)});return
        key=uuid.uuid4().hex;stream=body.get('stream',False);previous='';count=0;prompt=None;tokens=[];disconnected=False;started=False
        def send(value):
            nonlocal disconnected
            if disconnected:return
            try:self.wfile.write(('data: '+(value if isinstance(value,str) else json.dumps(value))+'\n\n').encode());self.wfile.flush()
            except (BrokenPipeError,ConnectionResetError):disconnected=True
        try:
            if stream:
                self.send_response(200);self.send_header('Content-Type','text/event-stream');self.send_header('Cache-Control','no-cache');self.end_headers();started=True
            for event in self.server.worker.generate(body,key):
                if event['event']=='token':
                    count=event['sampled_tokens'];prompt=event['prompt_tokens'];tokens.append(event['token_id'])
                    # Withhold a tokenizer's incomplete trailing UTF-8 replacement until complete.
                    text=event['text'].rstrip('\ufffd')
                    if not text.startswith(previous):raise ValueError('Tokenizer prefix revision cannot be streamed losslessly')
                    delta=text[len(previous):];previous=text
                    if stream:send({'id':key,'object':'chat.completion.chunk','choices':[{'index':0,'delta':{'content':delta},'finish_reason':None}],
                        'usage':{'prompt_tokens':prompt,'completion_tokens':count,'total_tokens':prompt+count},'token_id':event['token_id']})
                else:
                    if tokens!=event['generated_ids'] or len(event['prompt_ids'])!=prompt:raise ValueError('Worker final IDs disagree with streamed events')
                    full_text=event['text']
                    if not full_text.startswith(previous):raise ValueError('Final tokenizer text revises streamed prefix')
                    usage={'prompt_tokens':prompt,'completion_tokens':count,'total_tokens':prompt+count}
                    if stream:
                        send({'id':key,'choices':[{'index':0,'delta':{'content':full_text[len(previous):]},'finish_reason':event['finish_reason']}],'usage':usage,
                            'prompt_token_ids':event['prompt_ids'],'response_token_ids':tokens,'study':{'generation_mode':event.get('generation_mode'),'speculative_metrics':event.get('speculative_metrics')}});send('[DONE]')
                    else:self.reply(200,{'id':key,'choices':[{'index':0,'message':{'role':'assistant','content':full_text},'finish_reason':event['finish_reason'],
                        'prompt_token_ids':event['prompt_ids'],'response_token_ids':tokens}],'usage':usage,'study':{'generation_mode':event.get('generation_mode'),'speculative_metrics':event.get('speculative_metrics')}})
        except Exception as error:
            self.server.worker.ready=False
            if started:send({'error':repr(error)})
            else:self.reply(500,{'error':repr(error)})

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen38-27b-nvfp4','qwen35-4b-bf16','qwen38-27b-int4'],default='qwen38-27b-nvfp4');p.add_argument('--port',type=int,default=18082);p.add_argument('--log-root',type=Path,required=True);p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args()
    command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('Native worker command required')
    worker=Worker(command,a.log_root);server=ThreadingHTTPServer(('127.0.0.1',a.port),Handler);server.worker=worker;server.model=a.model
    try:server.serve_forever()
    finally:server.server_close();worker.close()
