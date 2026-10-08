"""Generate a marked private XQA adaptation from immutable upstream bytes.

Only the opt-in standalone native test uses this staged dependency. Inference
requires separate CUDA/native/compiled/default/memory/pretrained proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

PINNED_SUMS = 'e400f9df3446f52f3cb70f3fd0af812fa322bd25edf5c0c27e9070b23991bdb1'


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError('Pinned XQA adaptation boundary changed')
    return text.replace(old, new, 1)


def generate(upstream, output, allow_empty_build_directory=False):
    upstream, output = Path(upstream), Path(output)
    sums = (upstream / 'SHA256SUMS').read_bytes()
    if hashlib.sha256(sums).hexdigest() != PINNED_SUMS:
        raise ValueError('Unknown upstream XQA manifest')
    records = []
    for line in sums.decode().splitlines():
        expected, name = line.split('  ', 1)
        path = (upstream / name).resolve()
        if not path.is_relative_to(upstream.resolve()) or not path.is_file():
            raise ValueError('XQA dependency path changed')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Upstream XQA dependency changed: ' + name)
        records.append(name)
    if output.is_symlink() or output.exists():
        # Ninja precreates BYPRODUCT/OUTPUT parent directories. Only the
        # explicit private-build path may adopt a real, completely empty one.
        # Never repair or overwrite partial, corrupted or symlinked evidence.
        if (not allow_empty_build_directory or output.is_symlink() or
                not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(output)
    else:
        output.mkdir(parents=True)
    for name in records:
        shutil.copy2(upstream / name, output / name)
    prefix = '''// SPDX-License-Identifier: Apache-2.0
// GENERATED Garnet GPT-OSS adaptation of pinned FlashInfer XQA; original
// license/copyright follows. Changes: prefixed symbols, explicit checked
// per-context CUDA initialization, caller-owned shared-memory size, one-block
// launch and per-thread launch attributes. Original bytes remain in vendor tree.
'''
    source = (upstream / 'mha.cu').read_text(encoding='utf-8')
    initialization = '''static uint32_t configureKernel() {
  uint32_t size;
  cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize));
  cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, size);
  return size;
}

static uint32_t const hostSmemSize = configureKernel();'''
    source = replace_once(source, initialization, '''extern "C" cudaError_t GarnetGptOssXqaInitialize(uint32_t* dynamicSharedBytes) {
  if (!dynamicSharedBytes) return cudaErrorInvalidValue;
  *dynamicSharedBytes = 0;
  cudaError_t status = cudaMemcpyFromSymbol(dynamicSharedBytes, smemSize, sizeof(smemSize));
  if (status != cudaSuccess) return status;
  if (!*dynamicSharedBytes) return cudaErrorInvalidValue;
  return cudaFuncSetAttribute(kernel_mha, cudaFuncAttributeMaxDynamicSharedMemorySize, *dynamicSharedBytes);
}''')
    signature = 'void launchMHAFlashInfer(uint32_t multiProcessorCount,'
    changed = 'void launchMHAFlashInfer(uint32_t dynamicSharedBytes, uint32_t multiProcessorCount,'
    source = replace_once(source, signature, changed)
    start = source.index(changed)
    before, launcher = source[:start], source[start:]
    split = '''uint32_t const nbSubSeqPerSeq = [&]() -> uint32_t {
    if (!allowMultiBlockMode) {
      return 1;
    }
    return std::min<uint32_t>(std::max<uint32_t>(1U, multiProcessorCount / (batchSize * nbKHeads)),
                              divUp(maxSeqLen, ctaTile.x));
  }();'''
    launcher = replace_once(launcher, split, 'uint32_t const nbSubSeqPerSeq = 1; // Explicit Garnet single-block policy.')
    launcher = replace_once(launcher, 'makeLaunchConfig(dimGrid, dimCta, hostSmemSize, stream, enable_pdl)',
        'makeLaunchConfig(dimGrid, dimCta, dynamicSharedBytes, stream, enable_pdl)')
    symbols = '''#define launchMHA GarnetGptOssXqaUnusedLaunchMHA
#define launchMHAFlashInfer GarnetGptOssXqaLaunch
#define kernel_mha GarnetGptOssXqaKernel
#define smemSize GarnetGptOssXqaSmemSize
#define makeLaunchConfig GarnetGptOssXqaMakeLaunchConfig
'''
    (output / 'mha.cu').write_text(prefix + symbols + before + launcher, encoding='utf-8', newline='\n')
    header = (upstream / 'mha.h').read_text(encoding='utf-8')
    (output / 'mha.h').write_text(prefix + replace_once(header, signature, changed), encoding='utf-8', newline='\n')
    launch = (upstream / 'hostUtils.h').read_text(encoding='utf-8')
    launch = replace_once(launch, 'static cudaLaunchAttribute pdlAttr;', 'static thread_local cudaLaunchAttribute pdlAttr;')
    (output / 'hostUtils.h').write_text(prefix + launch, encoding='utf-8', newline='\n')
    manifest = dict(upstream_commit='946200de1ae94fc93fdd0926f0a13afd1fa7f0f1', upstream_manifest_sha256=PINNED_SUMS,
        scope='Generated source only; backend is not linked or GPU validated',
        generated_sha256={name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in records})
    (output / 'garnet-adaptation.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    return manifest


def verify_generated(upstream, output):
    """Reconstruct trusted bytes independently; never repair existing evidence."""
    output = Path(output)
    if output.is_symlink() or not output.is_dir():
        raise ValueError('Missing or symlinked generated XQA directory')
    with tempfile.TemporaryDirectory() as temporary:
        expected = Path(temporary) / 'expected'
        manifest = generate(upstream, expected)
        entries = list(output.iterdir())
        if (any(path.is_symlink() or not path.is_file() for path in entries) or
                {path.name for path in entries} != {path.name for path in expected.iterdir()}):
            raise ValueError('Generated XQA file set changed')
        for path in expected.iterdir():
            if (output / path.name).read_bytes() != path.read_bytes():
                raise ValueError('Generated XQA bytes changed: ' + path.name)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('upstream', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify-existing', action='store_true',
        help='Require exact reconstructed output bytes without modifying them')
    parser.add_argument('--allow-empty-build-directory', action='store_true',
        help='Private build only: accept a regular empty directory precreated by the build system')
    args = parser.parse_args()
    if args.verify_existing and args.allow_empty_build_directory:
        parser.error('Verification cannot adopt or repair output directories')
    result = (verify_generated(args.upstream, args.output) if args.verify_existing else
        generate(args.upstream, args.output, args.allow_empty_build_directory))
    print(json.dumps(result, indent=2))
