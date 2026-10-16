"""Fetch exact resolved comparator wheels through public PyPI with IPv4 curl.

Arguments: isolated root, evidence folder, package names. Requires the saved
offline `uv pip compile` requirements; does not install or run inference.
"""
import hashlib,json,platform,re,socket,subprocess,sys,time,urllib.request
from pathlib import Path
root=Path(sys.argv[1]);evidence=Path(sys.argv[2]);evidence.mkdir(parents=True,exist_ok=True)
allowed={'torch','llvmlite','scipy','mlx-metal','opencv-python','opencv-python-headless'}
requirements=(root/'downloads/requirements.txt').read_text();manifest=[]
edges=sorted({address[4][0] for address in socket.getaddrinfo('files.pythonhosted.org',443,socket.AF_INET)})
# The saved four-edge probe on this Mac measured this current DNS member fastest.
# It is only a per-curl connection choice; TLS/hostname and wheel hashes remain checked.
edges.sort(key=lambda ip:(ip!='151.101.64.223',ip))
try:
    for package in sys.argv[3:]:
        assert package in allowed,package
        version=re.search(r'^'+re.escape(package)+r'==([^\s;]+)',requirements,re.M)[1]
        with urllib.request.urlopen('https://pypi.org/pypi/'+package+'/'+version+'/json',timeout=60) as response:
            metadata=json.load(response)
        os_version=tuple(map(int,platform.mac_ver()[0].split('.')[:2]))
        def rank(asset):
            match=re.search(r'macosx_(\d+)_(\d+)_',asset['filename'])
            return tuple(map(int,match.groups()))+(1 if '-cp312-' in asset['filename'] else 0,)
        candidates=[a for a in metadata['urls'] if a['filename'].endswith('arm64.whl') and 'macosx_' in a['filename'] and
            ('-cp312-' in a['filename'] or '-cp37-abi3-' in a['filename'] or '-py3-none-' in a['filename']) and
            rank(a)[:2]<=os_version]
        candidates.sort(key=rank,reverse=True)
        assert candidates and (len(candidates)==1 or rank(candidates[0])>rank(candidates[1])),[a['filename'] for a in candidates]
        asset=candidates[0];assert asset['url'].startswith('https://files.pythonhosted.org/')
        path=root/'downloads'/asset['filename'];partial=path.with_suffix(path.suffix+'.curl-partial')
        started=time.monotonic()
        if not path.exists():
            print('Fetching exact pinned wheel',asset['filename'],asset['size'],flush=True)
            command=['/usr/bin/curl','-4','--http1.1','--fail','--location','--silent','--show-error',
                '--retry','0','--speed-limit','100000','--speed-time','20',
                '--connect-timeout','20','--max-time','900','--continue-at','-',
                '--output',str(partial),'--write-out','remote %{remote_ip}; downloaded %{size_download}; speed %{speed_download} bytes/sec; time %{time_total}\n',asset['url']]
            for attempt in range(8):
                edge=edges[attempt%len(edges)]
                result=subprocess.run(command[:-1]+['--resolve','files.pythonhosted.org:443:'+edge,command[-1]])
                if result.returncode==0:break
                print('Retry from saved bytes',partial.stat().st_size if partial.exists() else 0,'after curl exit',result.returncode,flush=True)
                time.sleep(1)
            result.check_returncode()
            assert partial.stat().st_size==asset['size'],asset['filename']
            assert hashlib.file_digest(partial.open('rb'),'sha256').hexdigest()==asset['digests']['sha256']
            partial.replace(path)
        digest=hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
        assert path.stat().st_size==asset['size'] and digest==asset['digests']['sha256']
        manifest.append(dict(package=package,version=version,filename=path.name,url=asset['url'],
            bytes=path.stat().st_size,sha256=digest,seconds=time.monotonic()-started))
        (evidence/'wheel_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print('Verified pinned wheel',path.name,flush=True)
finally:
    (evidence/'wheel_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
