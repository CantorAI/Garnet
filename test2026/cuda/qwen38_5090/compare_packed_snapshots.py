"""Compare all saved packed hidden/state F32 words outside the native VM."""
import argparse,array,hashlib,json,math,sys
from pathlib import Path

def compare_files(actual,expected,start=0,count=None):
 if actual.stat().st_size!=expected.stat().st_size or actual.stat().st_size%4:raise ValueError('F32 byte count mismatch')
 total=actual.stat().st_size//4
 if count is None:count=total-start
 if start<0 or count<1 or start+count>total:raise ValueError('Snapshot segment outside file')
 peak=0.;error2=0.;reference2=0.;words=0
 with actual.open('rb') as a,expected.open('rb') as b:
  a.seek(start*4);b.seek(start*4);remaining=count
  while remaining:
   raw=a.read(4*min(65536,remaining))
   if not raw:raise ValueError('Truncated snapshot')
   left=array.array('f');left.frombytes(raw);right=array.array('f');right.frombytes(b.read(len(raw)))
   if sys.byteorder!='little':left.byteswap();right.byteswap()
   for x,y in zip(left,right):
    if not math.isfinite(x) or not math.isfinite(y):raise ValueError('Nonfinite state snapshot')
    error=x-y;peak=max(peak,abs(error));error2+=error*error;reference2+=y*y;words+=1
   remaining-=len(left)
 relative=math.sqrt(error2/max(1e-20,reference2))
 return {'words':words,'max_absolute_error':peak,'relative_l2_error':relative,'passed':peak<=.05 and relative<=.01}

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();m=json.loads((a.root/'manifest.json').read_text());report={'chunks':[],'scope':'Full hidden/state word comparison across actual native programs; not model quality or timing'}
 if m['status']!='snapshots_completed':raise ValueError('Complete GPU snapshots required')
 segments=[('hidden',0,m['hidden_width'])]
 for layer in m['layout']:
  offset=m['hidden_width']+layer['offset']
  segments.extend([('layer'+str(layer['index'])+'-state1',offset,layer['state1_width']),('layer'+str(layer['index'])+'-state2',offset+layer['state1_width'],layer['state2_width'])])
 for chunk in m['chunks']:
  for snapshot in [chunk['reference']]+chunk['runs']:
   path=a.root/snapshot['file']
   if path.stat().st_size!=snapshot['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=snapshot['sha256']:raise ValueError('Snapshot digest mismatch')
  checks=[]
  for run in chunk['runs']:
   parts={name:compare_files(a.root/run['file'],a.root/chunk['reference']['file'],start,count) for name,start,count in segments}
   checks.append({'file':run['file'],'segments':parts,'passed':all(part['passed'] for part in parts.values())})
  report['chunks'].append({'active_tokens':chunk['active_tokens'],'runs':checks})
 report['passed']=all(r['passed'] for c in report['chunks'] for r in c['runs']);(a.root/'comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
 if not report['passed']:raise SystemExit(1)
