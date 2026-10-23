"""Merge verified official NPP/nvJPEG runtime wheels into the fixed CUDA SDK.

The versions match the CUDA13.2 SDK used by the local correctness build.
No existing differing SDK file is overwritten. Wheel staging is removed after
successful extraction; model files and deployments are never touched.
"""
import base64, csv, hashlib, io, json, subprocess, sys, zipfile
from pathlib import Path


def main():
    root = Path('D:/CantorAI/5090-study-20261023')
    target = root/'tools/cuda13'
    staging = root/'tools/wheel-staging'
    staging.mkdir(parents=True,exist_ok=True)
    versions = ('nvidia-npp==13.1.0.48','nvidia-nvjpeg==13.1.0.48')
    subprocess.run([sys.executable,'-m','pip','download','--no-input','--disable-pip-version-check',
                    '--no-cache-dir','--no-deps','--index-url','https://pypi.org/simple',
                    '--extra-index-url','https://pypi.nvidia.com','--dest',str(staging),*versions],check=True)
    files, wheels = [], []
    for package in ('nvidia_npp-13.1.0.48','nvidia_nvjpeg-13.1.0.48'):
        matches = list(staging.glob(package+'-*-win_amd64.whl'))
        if len(matches) != 1: raise ValueError('Expected one pinned Windows image SDK wheel')
        wheel = matches[0]
        with zipfile.ZipFile(wheel) as archive:
            records = [n for n in archive.namelist() if n.endswith('.dist-info/RECORD')]
            if len(records) != 1: raise ValueError('Missing wheel integrity record')
            checks = {r[0]:(r[1],r[2]) for r in csv.reader(io.StringIO(archive.read(records[0]).decode()))}
            for entry in archive.infolist():
                if entry.is_dir(): continue
                path = target/entry.filename
                if not path.resolve().is_relative_to(target.resolve()): raise ValueError('Unsafe wheel path')
                data = archive.read(entry)
                encoded, size = checks[entry.filename]
                if encoded:
                    algorithm, expected = encoded.split('=',1)
                    actual = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
                    if algorithm != 'sha256' or actual != expected or int(size) != len(data):
                        raise ValueError('Wheel member checksum mismatch')
                elif entry.filename != records[0]: raise ValueError('Unhashed wheel member')
                if path.exists() and path.read_bytes() != data:
                    raise ValueError('Differing existing SDK file: '+entry.filename)
                path.parent.mkdir(parents=True,exist_ok=True)
                path.write_bytes(data)
                files.append({'path':entry.filename,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()})
        wheels.append({'name':wheel.name,'sha256':hashlib.sha256(wheel.read_bytes()).hexdigest()})
    (root/'image-sdk-provenance.json').write_text(json.dumps({'requirements':versions,'wheels':wheels,'files':files},indent=2))
    for wheel in staging.glob('*.whl'):
        if wheel.name in {w['name'] for w in wheels}: wheel.unlink()
    print(json.dumps({'status':'installed-verified','files':len(files),'scope':'SDK artifacts; target compilation and inference still pending'}))


if __name__ == '__main__': main()
