"""Immediate JSON-line transport for the native study worker; no VM stdin wrapper."""
import json,os

MAX_FRAME_BYTES=2*1024*1024

def requests(read=None):
    read=read or (lambda:os.read(0,65536))
    pending=b''
    while True:
        block=read()
        if not block:
            if pending.strip():raise ValueError('Incomplete request frame at EOF')
            return
        pending+=block
        while b'\n' in pending:
            line,pending=pending.split(b'\n',1)
            if len(line)>MAX_FRAME_BYTES:raise ValueError('Request frame too large')
            value=json.loads(line)
            if not isinstance(value,dict):raise ValueError('Object request required')
            yield value
        if len(pending)>MAX_FRAME_BYTES:raise ValueError('Request frame too large')

def emit(value,write=None):
    write=write or (lambda raw:os.write(1,raw))
    raw=(json.dumps(value,separators=(',',':'),ensure_ascii=True)+'\n').encode()
    while raw:
        count=write(raw)
        if not isinstance(count,int) or count<1 or count>len(raw):raise OSError('Invalid worker pipe write')
        raw=raw[count:]
