"""Verify native layout evidence using complete outputs and frozen bounds."""
from pathlib import Path
import sys
source=Path(__file__).with_name('verify_direct_state_zero.py').read_text()
source=source.replace('direct-state-zero','decode-subsimd').replace('METAL_DIRECT_STATE_ZERO','METAL_SUBSIMD_DECODE')
source=source.replace("assert '100% tests passed' in ctest and 'metal_state_zero_reference' in ctest", "assert '100% tests passed' in ctest and 'metal_mps_matmul_reference' in ctest")
source=source.replace("assert 'state_zero checks 36 PASS' in (checks/'jobs/CTest-LastTest.log').read_text()\n",'')
source=source.replace("assert r['direct_state_zero_enabled']==(r['build_policy']=='candidate')", "assert r['subsimd_decode_enabled']==(r['build_policy']=='candidate')\n            assert not r['direct_state_zero_enabled']")
exec(compile(source,str(Path(__file__).with_suffix('.executed.py')),'exec'))
