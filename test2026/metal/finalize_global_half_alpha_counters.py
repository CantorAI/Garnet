"""Apply explicit global-alpha counter reset and successful publication once."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
for name,old,new,count in [
 ('src/backends/metal/metal_context.mm',b'impl_->packedPrefillNormDispatches=0;',b'impl_->packedPrefillNormDispatches=0;impl_->globalHalfAlphaDispatches=0;',2),
 ('src/backends/metal/decoder_session.cpp',b'packedPrefillNormDispatches_=0;',b'packedPrefillNormDispatches_=0;globalHalfAlphaDispatches_=0;',4),
 ('src/backends/metal/decoder_session.cpp',b'packedPrefillNormDispatches_=gpu_->LastPackedPrefillNormDispatches();',b'packedPrefillNormDispatches_=gpu_->LastPackedPrefillNormDispatches();globalHalfAlphaDispatches_=gpu_->LastGlobalHalfAlphaDispatches();',1),
]:
 path=root/name;raw=path.read_bytes();assert raw.count(old)==count and new not in raw;path.write_bytes(raw.replace(old,new))
