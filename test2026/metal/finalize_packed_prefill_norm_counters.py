"""Apply explicit packed-prefill counter resets/publication once."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
for name,old,new,count in [
 ('src/backends/metal/metal_context.mm',b'impl_->residualNormDispatches=0;',b'impl_->residualNormDispatches=0;impl_->packedPrefillNormDispatches=0;',2),
 ('src/backends/metal/decoder_session.cpp',b'residualNormDispatches_=0;',b'residualNormDispatches_=0;packedPrefillNormDispatches_=0;',4),
 ('src/backends/metal/decoder_session.cpp',b'residualNormDispatches_=gpu_->LastResidualNormDispatches();',b'residualNormDispatches_=gpu_->LastResidualNormDispatches();packedPrefillNormDispatches_=gpu_->LastPackedPrefillNormDispatches();',1),
]:
 path=root/name;raw=path.read_bytes();assert raw.count(old)==count and new not in raw;path.write_bytes(raw.replace(old,new))
