"""Actual CPU-only Ninja generation/repeat/tamper test; no CUDA compilation or GPU."""
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
repo=Path(__file__).resolve().parents[2];sys.path.insert(0,str(repo/'tools/gpt_oss'))
from prepare_xqa_sources import verify_generated
assert shutil.which('cmake') and shutil.which('ninja'), 'This gate requires actual CMake and Ninja'
upstream=repo/'plugins/gpt_oss/third_party/flashinfer_xqa';generator=repo/'tools/gpt_oss/prepare_xqa_sources.py'
with tempfile.TemporaryDirectory() as temporary:
 root=Path(temporary)
 for enabled in (False,True):
  source=root/('source-'+str(int(enabled)));source.mkdir();build=root/('build-'+str(int(enabled)))
  flag=' --allow-empty-build-directory' if enabled else ''
  (source/'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.18)
project(xqa_generation_cpu LANGUAGES NONE)
set(output "${CMAKE_CURRENT_BINARY_DIR}/nested/xqa")
add_custom_command(OUTPUT "${output}/garnet-adaptation.json"
 COMMAND "'''+Path(sys.executable).as_posix()+'" "'+generator.as_posix()+'" "'+upstream.as_posix()+'''" "${output}"'''+flag+'''
 VERBATIM)
add_custom_target(generate ALL DEPENDS "${output}/garnet-adaptation.json")
add_custom_target(verify COMMAND "'''+Path(sys.executable).as_posix()+'" "'+generator.as_posix()+'" "'+upstream.as_posix()+'''" "${output}" --verify-existing
 DEPENDS generate VERBATIM)
''',encoding='utf-8',newline='\n')
  result=subprocess.run(['cmake','-S',str(source),'-B',str(build),'-G','Ninja'],capture_output=True,text=True)
  assert result.returncode==0,result.stdout+result.stderr
  def run():return subprocess.run(['cmake','--build',str(build),'--target','verify'],capture_output=True,text=True)
  result=run();output=build/'nested/xqa'
  if not enabled:
   assert result.returncode!=0 and 'FileExistsError' in result.stdout+result.stderr
   assert output.is_dir() and not list(output.iterdir())
  else:
   assert result.returncode==0,result.stdout+result.stderr
   manifest=verify_generated(upstream,output)
   assert run().returncode==0 and verify_generated(upstream,output)==manifest
   victim=output/'mha.cu';bad=victim.read_bytes()+b'\n// retained tamper\n';victim.write_bytes(bad)
   result=run();assert result.returncode!=0 and 'Generated XQA bytes changed' in result.stdout+result.stderr
   assert victim.read_bytes()==bad
print('Actual CPU Ninja: original empty-directory failure reproduced, explicit empty adoption/repeat pass, changed evidence rejected without overwrite; no native/GPU proof')
