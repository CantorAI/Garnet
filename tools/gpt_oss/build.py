"""Build Garnet against an existing XLang3 runtime without rebuilding XLang3.

Windows: run in a Visual Studio x64 developer shell. Linux: use CUDA + TensorRT
SDKs matching the selected GPU. All build products stay under --build-dir.
"""
import argparse
import os
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--runtime-executable', type=Path, required=True)
parser.add_argument('--runtime-library', type=Path, required=True, help='Windows import .lib or Linux shared .so')
parser.add_argument('--runtime-dll', type=Path, help='Windows runtime DLL')
parser.add_argument('--tensorrt-root', type=Path, required=True)
parser.add_argument('--build-dir', type=Path, required=True)
parser.add_argument('--cuda-architectures', default='native')
parser.add_argument('--enable-nccl', action='store_true',
                    help='build GPT-OSS TP2 collectives and run their two-GPU parity test (Linux)')
parser.add_argument('--cmake', default='cmake')
parser.add_argument('--jobs', type=int, default=4)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
build = args.build_dir.resolve()
executable, library = args.runtime_executable.resolve(), args.runtime_library.resolve()
assert executable.is_file() and library.is_file()
shared = args.runtime_dll.resolve() if args.runtime_dll else library
assert shared.is_file()
if library.suffix.lower() == '.lib':
    assert args.runtime_dll, '--runtime-dll is required with a Windows import library'
source = build / 'source'
source.mkdir(parents=True, exist_ok=True)


def quote(path):
    # Bracket quoting keeps spaces, backslashes and semicolons out of CMake parsing.
    value = path.as_posix()
    assert ']===]' not in value
    return '[===[' + value + ']===]'


text = '''cmake_minimum_required(VERSION 3.24)
project(garnet_plugin_check LANGUAGES CXX CUDA)
enable_testing()
add_library(xlang3_runtime SHARED IMPORTED GLOBAL)
add_executable(xlang3 IMPORTED GLOBAL)
'''
text += 'set_target_properties(xlang3_runtime PROPERTIES IMPORTED_LOCATION ' + quote(shared)
if library.suffix.lower() == '.lib':
    text += ' IMPORTED_IMPLIB ' + quote(library)
text += ')\nset_target_properties(xlang3 PROPERTIES IMPORTED_LOCATION ' + quote(executable) + ')\n'
text += 'add_subdirectory(' + quote(repo) + ' Garnet)\n'
(source / 'CMakeLists.txt').write_text(text)
configure = [args.cmake, '-S', str(source), '-B', str(build), '-G', 'Ninja',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CUDA_ARCHITECTURES=' + args.cuda_architectures,
    '-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=' + (build / 'bin').as_posix(),
    '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + (build / 'bin').as_posix(),
    '-DGARNET_TENSORRT_ROOT=' + args.tensorrt_root.resolve().as_posix()]
if args.enable_nccl:
    configure.append('-DGARNET_GPT_OSS_ENABLE_NCCL=ON')
subprocess.run(configure, check=True)
targets = ['garnet', 'garnet_gpt_oss_kernel_parity']
if args.enable_nccl:
    targets.append('garnet_gpt_oss_tp_collective_test')
subprocess.run([args.cmake, '--build', str(build), '--target', *targets,
                '--parallel', str(args.jobs)], check=True)
if args.enable_nccl:
    subprocess.run([args.cmake, '--build', str(build), '--target', 'garnet_gpt_oss_ops',
                    '--parallel', str(args.jobs)], check=True)
    subprocess.run([args.cmake, '-E', 'env',
        'LD_LIBRARY_PATH=' + str(build / 'bin') + ':' + str(args.tensorrt_root.resolve() / 'lib'),
        'ctest', '--test-dir', str(build), '--output-on-failure',
        '-R', '^garnet_gpt_oss_tp_collective_test$'], check=True)
    environment = os.environ.copy()
    environment['LD_LIBRARY_PATH'] = os.pathsep.join([
        str(build / 'bin'), str(args.tensorrt_root.resolve() / 'lib'),
        environment.get('LD_LIBRARY_PATH', '')])
    subprocess.run([str(executable), str(repo / 'test2026/gpt_oss/tp_collective_graph.py'),
                    str(build / 'tp2-collective-xmodel-test')],
                   cwd=repo, env=environment, check=True)
for artifact in (executable, shared):
    destination = build / 'bin' / artifact.name
    if artifact.resolve() != destination.resolve():
        shutil.copy2(artifact, destination)
print('Built runtime and operator plugin:', build / 'bin')
