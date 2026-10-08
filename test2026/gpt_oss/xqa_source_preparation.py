"""Pinned-source/generation guard; no CUDA execution or backend selection."""
import hashlib
from pathlib import Path
import shutil
import sys
import tempfile

repo=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(repo/'tools/gpt_oss'))
from prepare_xqa_sources import generate, verify_generated

upstream=repo/'plugins/gpt_oss/third_party/flashinfer_xqa'
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary)
    first=generate(upstream,root/'first')
    second=generate(upstream,root/'second')
    assert verify_generated(upstream,root/'first') == first
    assert (root/'first/garnet-adaptation.json').read_bytes() == (root/'second/garnet-adaptation.json').read_bytes()
    assert b'\r' not in (root/'first/garnet-adaptation.json').read_bytes()
    assert first==second
    assert len(first['generated_sha256']) == 17
    original=(upstream/'mha.cu').read_bytes()
    assert hashlib.sha256(original).hexdigest()=='fac9c9a116be125b800257c455ee3abe2a2a620b2816a3c254792d3a5033ace4'
    source=(root/'first/mha.cu').read_text()
    assert 'static uint32_t const hostSmemSize = configureKernel();' not in source
    assert 'extern "C" cudaError_t GarnetGptOssXqaInitialize' in source
    assert 'if (status != cudaSuccess) return status;' in source
    assert '#define kernel_mha GarnetGptOssXqaKernel' in source
    launcher=source[source.index('void launchMHAFlashInfer'):]
    assert 'nbSubSeqPerSeq = 1;' in launcher
    assert 'multiProcessorCount / (batchSize * nbKHeads)' not in launcher
    assert 'makeLaunchConfig(dimGrid, dimCta, dynamicSharedBytes' in launcher
    assert 'static thread_local cudaLaunchAttribute' in (root/'first/hostUtils.h').read_text()
    for name in first['generated_sha256']:
        assert (root/'first'/name).read_bytes()==(root/'second'/name).read_bytes()
    assert (root/'first/LICENSE').read_bytes()==(upstream/'LICENSE').read_bytes()
    assert (root/'first/NOTICE').read_bytes()==(upstream/'NOTICE').read_bytes()
    assert hashlib.sha256((upstream/'NOTICE').read_bytes()).hexdigest()=='90bb9e1dec06f26a34f8f8ce98c53ded10b4fd8e1a9e91e2360e6be354e81db3'
    try: generate(upstream,root/'first')
    except FileExistsError: pass
    else: raise AssertionError('Existing generated evidence was overwritten')
    for name in ('mha.cu', 'mha.h', 'hostUtils.h', 'NOTICE', 'garnet-adaptation.json'):
        output = root / ('tamper-' + name)
        shutil.copytree(root/'first', output)
        changed_bytes = (output/name).read_bytes() + b'\n// tampered generated evidence\n'
        (output/name).write_bytes(changed_bytes)
        try: verify_generated(upstream, output)
        except ValueError: pass
        else: raise AssertionError('Modified generated output accepted: ' + name)
        assert (output/name).read_bytes() == changed_bytes
    for extra in ('unexpected.cu', 'unexpected-directory'):
        output = root / extra
        shutil.copytree(root/'first', output)
        if extra.endswith('.cu'): (output/extra).write_bytes(b'extra')
        else: (output/extra).mkdir()
        try: verify_generated(upstream, output)
        except ValueError: pass
        else: raise AssertionError('Extra generated entry accepted')
        assert (output/extra).exists()
    output = root/'missing'; shutil.copytree(root/'first', output)
    (output/'mha.h').unlink()
    try: verify_generated(upstream, output)
    except ValueError: pass
    else: raise AssertionError('Missing generated output accepted')
    assert not (output/'mha.h').exists()
    # Windows may lack symlink privilege. Linux must exercise this rejection.
    output = root/'symlink'; shutil.copytree(root/'first', output)
    (output/'mha.h').unlink()
    try: (output/'mha.h').symlink_to(root/'first/mha.h')
    except OSError:
        if sys.platform != 'win32': raise
        print('Windows symlink creation unavailable; Linux test requires this negative control')
    else:
        try: verify_generated(upstream, output)
        except ValueError: pass
        else: raise AssertionError('Symlinked generated output accepted')
        assert (output/'mha.h').is_symlink()
    root_link = root/'root-link'
    try: root_link.symlink_to(root/'first', target_is_directory=True)
    except OSError:
        if sys.platform != 'win32': raise
    else:
        try: verify_generated(upstream, root_link)
        except ValueError: pass
        else: raise AssertionError('Symlinked generated root accepted')
        assert root_link.is_symlink()
    changed=root/'changed';shutil.copytree(upstream,changed)
    (changed/'mha.cu').write_bytes(original+b'\n// changed upstream\n')
    try: generate(changed,root/'bad-source')
    except ValueError: assert not (root/'bad-source').exists()
    else: raise AssertionError('Mutated upstream source accepted')
    changed_notice=root/'changed-notice';shutil.copytree(upstream,changed_notice)
    (changed_notice/'NOTICE').write_bytes(b'Changed upstream notice')
    try: generate(changed_notice,root/'bad-notice')
    except ValueError: assert not (root/'bad-notice').exists()
    else: raise AssertionError('Mutated upstream NOTICE accepted')
    (changed/'SHA256SUMS').write_text('untrusted replacement manifest')
    try: generate(changed,root/'bad-manifest')
    except ValueError: assert not (root/'bad-manifest').exists()
    else: raise AssertionError('Unpinned source manifest accepted')
print('Pinned17-file dependency, retained LICENSE/NOTICE, deterministic LF adaptation and source/notice/overwrite/exact generated-byte/file-set/available symlink guards passed; no GPU proof')
