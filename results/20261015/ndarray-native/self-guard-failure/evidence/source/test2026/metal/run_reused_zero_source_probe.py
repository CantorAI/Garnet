"""Exclusive cached-zero-source component, same dependent read and exact gates."""
from pathlib import Path
import subprocess,sys
folder=Path(__file__).parent
subprocess.run([sys.executable,str(folder/'make_reused_zero_source_probe.py'),str(folder/'reused_zero_source_gpu.mm')],check=True)
source=(folder/'run_joint_state_zero_probe.py').read_text().replace('joint_state_zero_gpu.mm','reused_zero_source_gpu.mm').replace('joint_state_zero_gpu','reused_zero_source_gpu')
source=source.replace("paths=[Path(__file__),", "paths=[s/'make_reused_zero_source_probe.py',s/'joint_state_zero_gpu.mm',s/'run_joint_state_zero_probe.py',Path(__file__),")
source=source.replace('cpu_memset_then_gpu_read','retained_zero_vector_memcpy_then_gpu_read').replace('joint_over_direct_cpu_wall','joint_over_reused_cpu_wall')
source=source.replace("joint_over_legacy_wall=means['2']['wall_ms']/means['0']['wall_ms'],", "reused_over_legacy_wall=means['1']['wall_ms']/means['0']['wall_ms'],joint_over_legacy_wall=means['2']['wall_ms']/means['0']['wall_ms'],")
source=source.replace("configs=configs,profiles=profiles,", "source_allocation='One max-state FP32 zero vector per profile before timing; initialization cost excluded; original memcpy loop unchanged',configs=configs,profiles=profiles,")
exec(compile(source,__file__,'exec'))
