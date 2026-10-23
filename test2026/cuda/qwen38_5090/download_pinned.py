"""Download public pinned artifacts from official CDN URLs, verifying primary LFS hashes.

The temporary transport manifest may contain expiring public CDN signatures.
It is deleted after loading and must never be committed or copied into a paper.
Permanent provenance records only model/revision, file sizes and SHA-256 values.
"""
import argparse, concurrent.futures, hashlib, json, os, shutil, threading, time
from pathlib import Path
from urllib.request import Request, urlopen

MODEL = 'nvidia/Qwen3.8-27B-NVFP4'
REVISION = '482ca0f3832238542f8f5295dde86b5f22711d80'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''): h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manifest',type=Path)
    parser.add_argument('output',type=Path)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    args.manifest.unlink()
    if manifest['model'] != MODEL or manifest['revision'] != REVISION:
        raise ValueError('Only the verified pinned checkpoint is admitted')
    root = args.output
    root.mkdir(parents=True,exist_ok=True)
    total = sum(item['bytes'] for item in manifest['files'])
    if total > 24*1024**3: raise ValueError('Manifest exceeds the initial download budget')
    if shutil.disk_usage(root).free < total+2*1024**3:
        raise ValueError('Insufficient free disk for this single checkpoint')
    progress, completed, lock = {}, [], threading.Lock()
    def record(name, size, status):
        with lock:
            progress[name] = {'bytes':size,'status':status}
            temp = root/'download-progress.tmp'
            temp.write_text(json.dumps({'model':MODEL,'revision':REVISION,'files':progress},indent=2))
            os.replace(temp,root/'download-progress.json')
    def download(item):
        name = item['name']
        if Path(name).name != name or name in ('.','..'): raise ValueError('Unsafe artifact filename')
        expected, size = item['sha256'], item['bytes']
        if len(expected) != 64 or size < 1: raise ValueError('Invalid artifact digest/size')
        target = root/name
        if target.is_file() and target.stat().st_size == size and digest(target) == expected:
            record(name,size,'verified-existing')
            return {'name':name,'bytes':size,'sha256':expected}
        partial = root/(name+'.partial')
        for attempt in range(3):
            try:
                offset = partial.stat().st_size if partial.exists() else 0
                if offset > size: raise ValueError('Partial file exceeds declared artifact size')
                if offset == size:
                    if digest(partial) != expected: raise ValueError('Completed partial digest mismatch')
                    os.replace(partial,target)
                    record(name,size,'verified')
                    return {'name':name,'bytes':size,'sha256':expected}
                request = Request(item['url'],headers={'Range':'bytes='+str(offset)+'-'} if offset else {})
                with urlopen(request,timeout=60) as response:
                    if offset and response.status != 206: raise ValueError('Server did not honor resume range')
                    if offset and not response.headers.get('Content-Range','').startswith('bytes '+str(offset)+'-'):
                        raise ValueError('Resume range starts at wrong offset')
                    if response.status not in (200,206): raise ValueError('Unexpected download status')
                    record(name,offset,'downloading')
                    h = hashlib.sha256()
                    if offset:
                        with partial.open('rb') as old:
                            for block in iter(lambda:old.read(4*1024*1024),b''): h.update(block)
                    count, last = offset, time.monotonic()
                    with partial.open('ab' if offset else 'wb') as f:
                        while True:
                            block = response.read(4*1024*1024)
                            if not block: break
                            if count+len(block) > size: raise ValueError('Server exceeded artifact size')
                            f.write(block); h.update(block); count += len(block)
                            if time.monotonic()-last >= 5:
                                record(name,count,'downloading'); last=time.monotonic()
                    if count != size or h.hexdigest() != expected:
                        raise ValueError('Artifact byte count or SHA-256 mismatch')
                os.replace(partial,target)
                record(name,size,'verified')
                return {'name':name,'bytes':size,'sha256':expected}
            except Exception as error:
                # Do not print signed URLs embedded in HTTP exception strings.
                record(name,partial.stat().st_size if partial.exists() else 0,'retry-'+str(attempt+1)+':'+type(error).__name__)
                if attempt == 2: raise RuntimeError('Download failed for '+name+': '+type(error).__name__) from None
                time.sleep(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        for result in pool.map(download,manifest['files']): completed.append(result)
    provenance = {'model':MODEL,'revision':REVISION,'files':completed,
                  'primary_metadata_source':'https://huggingface.co/api/models/'+MODEL+'/revision/'+REVISION,
                  'scope':'Artifact bytes verified; inference and performance unverified'}
    (root/'download-provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps({'status':'verified','files':len(completed),'bytes':sum(x['bytes'] for x in completed)}))


if __name__ == '__main__': main()
