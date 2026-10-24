"""Immediate JSON-line transport for the native study worker; no VM stdin wrapper."""
import json,os

MAX_FRAME_BYTES=2*1024*1024

def transport_text(text):
    raw=text.encode('utf8')
    try:return raw.decode('utf8','strict')
    except UnicodeDecodeError:pass
    # Consume maximal valid UTF-8 subparts rather than replacing every byte.
    parts=[];i=0
    while i<len(raw):
        first=raw[i]
        if first<128:parts.append(raw[i:i+1]);i+=1;continue
        if 194<=first<=223:size=2;lower=128;upper=191
        elif 224<=first<=239:
            size=3;lower=160 if first==224 else 128;upper=159 if first==237 else 191
        elif 240<=first<=244:
            size=4;lower=144 if first==240 else 128;upper=143 if first==244 else 191
        else:parts.append(b'\xef\xbf\xbd');i+=1;continue
        used=1
        while used<size and i+used<len(raw):
            value=raw[i+used]
            if not ((lower<=value<=upper) if used==1 else (128<=value<=191)):break
            used+=1
        if used==size:parts.append(raw[i:i+used])
        else:parts.append(b'\xef\xbf\xbd')
        i+=used
    return b''.join(parts).decode('utf8','strict')


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
    if isinstance(value.get('text'),str):
        # The native tokenizer can return an incomplete UTF-8 byte prefix.
        # Normalize transport text; token IDs/counts and the caller stay intact.
        value=dict(value)
        value['text']=transport_text(value['text'])
    raw=(json.dumps(value,separators=(',',':'),ensure_ascii=True)+'\n').encode()
    while raw:
        count=write(raw)
        if not isinstance(count,int) or count<1 or count>len(raw):raise OSError('Invalid worker pipe write')
        raw=raw[count:]
