"""Install verified stable Apple Silicon comparator wheels in a private environment.

Arguments: isolated installation directory, evidence directory.
No shell installer, global package changes, model download, or inference.
"""
import hashlib,json,os,platform,re,subprocess,sys,tarfile,urllib.request,urllib.parse
from pathlib import Path

root=Path(sys.argv[1]).resolve();evidence=Path(sys.argv[2]).resolve()
root.mkdir(parents=True,exist_ok=True);evidence.mkdir(parents=True,exist_ok=True)
manifest={'platform':platform.platform(),'commands':[],'downloads':[]}
cached_path=root/'downloads/release_metadata.json'
cached=json.loads(cached_path.read_text()) if cached_path.exists() else None

def fetch_json(url):
    if cached and url in cached['sources']:return cached['sources'][url]
    with urllib.request.urlopen(url,timeout=90) as response:return json.load(response)

def download(asset):
    path=root/'downloads'/asset['name'];path.parent.mkdir(parents=True,exist_ok=True)
    expected=asset.get('digest');assert expected and expected.startswith('sha256:'),asset['name']
    if not path.exists():
        partial=path.with_suffix(path.suffix+'.partial')
        print('Downloading',asset['name'],asset['size'],flush=True)
        mirror=os.environ.get('GARNET_PUBLIC_ASSET_MIRROR')
        url=mirror.rstrip('/')+'/'+urllib.parse.quote(asset['name']) if mirror else asset['browser_download_url']
        with urllib.request.urlopen(url,timeout=180) as response,partial.open('wb') as stream:
            while block:=response.read(1024*1024):stream.write(block)
        partial.replace(path)
    digest='sha256:'+hashlib.file_digest(path.open('rb'),'sha256').hexdigest()
    assert path.stat().st_size==asset['size'] and digest==expected,asset['name']
    manifest['downloads'].append(dict(name=path.name,url=asset['browser_download_url'],size=path.stat().st_size,digest=digest,
        asset_mirror=os.environ.get('GARNET_PUBLIC_ASSET_MIRROR')))
    return path

def run(command,env=None):
    print('Running',json.dumps([str(v) for v in command]),flush=True)
    status=subprocess.run([str(v) for v in command],env=env)
    manifest['commands'].append(dict(command=[str(v) for v in command],exit_code=status.returncode))
    status.check_returncode()

try:
    assert platform.system()=='Darwin' and platform.machine()=='arm64'
    uv_release=fetch_json('https://api.github.com/repos/astral-sh/uv/releases/latest')
    uv_asset=next(a for a in uv_release['assets'] if a['name']=='uv-aarch64-apple-darwin.tar.gz')
    archive=download(uv_asset);uv=root/'bin/uv';uv.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(archive) as bundle:
        member=next(m for m in bundle.getmembers() if m.isfile() and Path(m.name).name=='uv')
        uv.write_bytes(bundle.extractfile(member).read());uv.chmod(0o755)
    env=os.environ.copy();env['UV_CACHE_DIR']=str(root/'uv-cache');env['UV_PYTHON_INSTALL_DIR']=str(root/'python')
    env['UV_PYTHON_BIN_DIR']=str(root/'python-bin')
    run([uv,'python','install','3.12','--no-bin'],env)
    if not (root/'venv/bin/python').exists():run([uv,'venv','--python','3.12',root/'venv'],env)
    release=fetch_json('https://api.github.com/repos/vllm-project/vllm-metal/releases/latest')
    tag=release['tag_name'];assert re.fullmatch(r'v\d+\.\d+\.\d+',tag),tag
    wheel=download(next(a for a in release['assets'] if a['name'].endswith('cp312-cp312-macosx_15_0_arm64.whl')))
    if cached:core_tag=cached['core_release_tag']
    else:
        with urllib.request.urlopen('https://raw.githubusercontent.com/vllm-project/vllm-metal/'+tag+'/.github/vllm-release-tag.commit',timeout=60) as response:
            core_tag=response.read().decode().strip()
    assert re.fullmatch(r'v\d+\.\d+\.\d+',core_tag),core_tag
    core=fetch_json('https://api.github.com/repos/vllm-project/vllm/releases/tags/'+core_tag)
    core_name='vllm-'+core_tag[1:]+'+cpu-cp312-cp312-macosx_11_0_arm64.whl'
    core_asset=next(a for a in core['assets'] if a['name']==core_name)
    if (root/'downloads'/core_name).exists():core_wheel=download(core_asset)
    else:
        assert core_asset['digest'].startswith('sha256:')
        # uv's Rust downloader works where Python's GitHub redirect transfer
        # does not; the URL hash is checked during installation.
        core_wheel=core_asset['browser_download_url']+'#sha256='+core_asset['digest'].split(':',1)[1]
        manifest['downloads'].append(dict(name=core_name,url=core_asset['browser_download_url'],size=core_asset['size'],
            digest=core_asset['digest'],verification='uv direct-URL SHA256 constraint'))
    manifest.update(uv_version=uv_release['tag_name'],metal_release=tag,core_release=core_tag)
    run([uv,'pip','install','--python',root/'venv/bin/python',core_wheel,wheel],env)
    run([root/'venv/bin/python','-c','import importlib.metadata as m,platform; print(platform.machine()); print({n:m.version(n) for n in ["vllm","vllm-metal","mlx","mlx-lm","torch"]})'],env)
    run([root/'venv/bin/python','-c','import vllm,vllm_metal,torch; import mlx.core as mx; assert mx.metal.is_available(); print("Comparator imports passed; Metal device", mx.metal.device_info()["device_name"])'],env)
    with (evidence/'packages.json').open('w') as stream:
        subprocess.run([str(uv),'pip','list','--python',str(root/'venv/bin/python'),'--format','json'],env=env,stdout=stream,check=True)
    manifest['exit_code']=0
except Exception as error:
    manifest.update(exit_code=1,error_type=type(error).__name__);raise
finally:
    (evidence/'setup_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
