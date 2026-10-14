"""Apply explicit counter reset/publication sites once, preserving source bytes."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
for name,old,new,count in [
 ('src/backends/metal/metal_context.mm',b'impl_->constantHeadDispatches=0;',b'impl_->constantHeadDispatches=0;impl_->residualNormDispatches=0;',2),
 ('src/backends/metal/decoder_session.cpp',b'constantHeadDispatches_=0;',b'constantHeadDispatches_=0;residualNormDispatches_=0;',4),
 ('src/backends/metal/decoder_session.cpp',b'constantHeadDispatches_=gpu_->LastConstantHeadDispatches();',b'constantHeadDispatches_=gpu_->LastConstantHeadDispatches();residualNormDispatches_=gpu_->LastResidualNormDispatches();',1),
]:
 path=root/name;raw=path.read_bytes();assert raw.count(old)==count and new not in raw;path.write_bytes(raw.replace(old,new))
