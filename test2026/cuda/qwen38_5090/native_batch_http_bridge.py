"""Research fixed-batch dispatcher; CPU protocol tests are not model benchmarks.

A compatible native worker must announce study_batch_v1 and its physical capacity.
Only synchronized worker token events are routed to clients; none are synthesized.
"""
import argparse,json,queue,threading,time
from pathlib import Path
from native_http_bridge import Worker,Handler,ThreadingHTTPServer

class Ticket:
    def __init__(self,key,body):
        self.key=key;self.body=body;self.events=queue.Queue();self.done=threading.Event()

class BatchWorker(Worker):
    def __init__(self,command,root,batch=2,wait_ms=2):
        if type(batch) is not int or batch not in (2,4) or not 0<=wait_ms<=20:
            raise ValueError('Fixed study batch2/4 and bounded grouping delay required')
        self.capacity=batch;self.wait_seconds=wait_ms/1000
        self.tickets={};self.ticket_lock=threading.Lock();self.jobs=queue.Queue(maxsize=64);self.stop=threading.Event()
        super().__init__(command,root)
        self.dispatcher=threading.Thread(target=self._dispatch,daemon=True);self.dispatcher.start()

    def _fail(self,error):
        self.failure=str(error);self.ready=False;self.stop.set()
        with self.ticket_lock:
            for ticket in self.tickets.values():
                ticket.events.put({'event':'worker_exit','error':self.failure});ticket.done.set()

    def _read(self):
        try:
            with (self.root/'worker-stdout.jsonl').open('a',encoding='utf8') as raw:
                for line in self.process.stdout:
                    now=time.monotonic();text=line.decode('utf8',errors='replace').rstrip('\r\n')
                    raw.write(json.dumps({'received':now,'line':text})+'\n');raw.flush()
                    try:event=json.loads(text)
                    except ValueError:continue
                    if not isinstance(event,dict):continue
                    kind=event.get('event')
                    if kind=='ready':
                        if event.get('protocol')!='study_batch_v1' or event.get('batch')!=self.capacity:
                            raise ValueError('Worker did not announce the required batch protocol/capacity')
                        self.ready_info=event;self.ready=True
                    elif kind in ('token','completed','error'):
                        with self.ticket_lock:ticket=self.tickets.get(event.get('id'))
                        if ticket is None:raise ValueError('Worker emitted an unknown request ID')
                        ticket.events.put(event)
                        if kind in ('completed','error'):ticket.done.set()
        except Exception as error:self._fail(error)
        finally:self._fail(self.failure or 'Batch worker exited')

    def _dispatch(self):
        try:
            while not self.stop.is_set():
                try:first=self.jobs.get(timeout=.1)
                except queue.Empty:continue
                if first is None:break
                group=[first];deadline=time.monotonic()+self.wait_seconds
                while len(group)<self.capacity:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:break
                    try:ticket=self.jobs.get(timeout=remaining)
                    except queue.Empty:break
                    if ticket is None:self.stop.set();break
                    group.append(ticket)
                if self.stop.is_set():break
                if not self.healthy():raise RuntimeError('Native batch worker is not ready')
                wire=(json.dumps({'batch':[{'id':t.key,'body':t.body} for t in group]},separators=(',',':'))+'\n').encode()
                with (self.root/'batch-dispatch.jsonl').open('a',encoding='utf8') as log:
                    log.write(json.dumps({'monotonic':time.monotonic(),'physical_capacity':self.capacity,
                        'request_ids':[t.key for t in group],'real_rows':len(group)})+'\n')
                while wire:
                    written=self.process.stdin.write(wire)
                    if not written:raise OSError('Batch request pipe closed')
                    wire=wire[written:]
                self.process.stdin.flush()
                deadline=time.monotonic()+300
                for ticket in group:
                    if not ticket.done.wait(max(0,deadline-time.monotonic())):
                        raise TimeoutError('Native batch event deadline')
        except Exception as error:self._fail(error)

    def generate(self,body,key):
        if not self.healthy():raise RuntimeError('Native batch worker is not ready')
        ticket=Ticket(key,body)
        with self.ticket_lock:
            if key in self.tickets:raise ValueError('Duplicate active request ID')
            self.tickets[key]=ticket
        try:
            self.jobs.put(ticket,timeout=1);count=0
            while True:
                try:event=ticket.events.get(timeout=300)
                except queue.Empty:raise TimeoutError('Client batch event deadline')
                if event['event']=='worker_exit':raise RuntimeError(event['error'])
                if event.get('id')!=key:raise ValueError('Batch request ID mismatch')
                if event['event']=='error':raise RuntimeError(event['error'])
                if event['event']=='token':
                    count+=1
                    if event['sampled_tokens']!=count:raise ValueError('Batch token count mismatch')
                elif event['event']=='completed':
                    if len(event['generated_ids'])!=count:raise ValueError('Batch final token count mismatch')
                yield event
                if event['event']=='completed':return
        finally:
            with self.ticket_lock:self.tickets.pop(key,None)

    def close(self):
        self.stop.set()
        try:self.jobs.put_nowait(None)
        except queue.Full:pass
        self.dispatcher.join(timeout=120)
        if self.dispatcher.is_alive():raise TimeoutError('Batch dispatcher did not retire')
        super().close()

class BatchHandler(Handler):
    def do_GET(self):
        if self.path!='/health':self.reply(404,{'error':'Unknown study endpoint'});return
        worker=self.server.worker;healthy=worker.healthy()
        info={'research_only':True,'batch':worker.capacity,'context_capacity':4096,'queued_concurrency':True,
              'grouping_delay_ms':worker.wait_seconds*1000,'gpu_batching_qualification':'pending'}
        if healthy:info['native_worker']=worker.ready_info
        self.reply(200 if healthy else 503,info)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18082);p.add_argument('--log-root',type=Path,required=True)
    p.add_argument('--batch',type=int,choices=[2,4],default=2);p.add_argument('--wait-ms',type=float,default=2)
    p.add_argument('command',nargs=argparse.REMAINDER);a=p.parse_args();command=a.command[1:] if a.command[:1]==['--'] else a.command
    if not command:p.error('Compatible native batch worker command required')
    worker=BatchWorker(command,a.log_root,a.batch,a.wait_ms);server=ThreadingHTTPServer(('127.0.0.1',a.port),BatchHandler)
    server.worker=worker;server.model='qwen38-27b-nvfp4'
    try:server.serve_forever()
    finally:server.server_close();worker.close()
