"""Add actual optional MLP counters alongside existing global-alpha telemetry."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
assert 'joinedGlobalMlpBlocks_' not in (root/'src/backends/metal/decoder_session.h').read_text(encoding='utf-8'),'Counter preparation was already applied; do not modify the completed source again'
def edit(name,changes):
 p=root/name;s=p.read_text(encoding='utf-8')
 for before,after in changes:
  assert before in s,(name,before);s=s.replace(before,after)
 p.write_bytes(s.encode())
edit('src/backends/metal/metal_context.mm',[('impl_->globalHalfAlphaDispatches=0;','impl_->globalHalfAlphaDispatches=0;impl_->joinedGlobalMlpBlocks=0;')])
edit('src/backends/metal/decoder_session.h',[
 ('size_t globalHalfAlphaDispatches_=0;','size_t globalHalfAlphaDispatches_=0;\n    size_t joinedGlobalMlpBlocks_=0;'),
 ('size_t LastGlobalHalfAlphaDispatches() const {return globalHalfAlphaDispatches_;}','size_t LastGlobalHalfAlphaDispatches() const {return globalHalfAlphaDispatches_;}\n    size_t LastJoinedGlobalMlpBlocks() const {return joinedGlobalMlpBlocks_;}'),
 ('APISET().AddProp("global_half_alpha_dispatches",&MetalDecoderSession::LastGlobalHalfAlphaDispatches);','APISET().AddProp("global_half_alpha_dispatches",&MetalDecoderSession::LastGlobalHalfAlphaDispatches);\n        APISET().AddProp("joined_global_mlp_blocks",&MetalDecoderSession::LastJoinedGlobalMlpBlocks);')])
edit('src/backends/metal/decoder_session.cpp',[
 ('globalHalfAlphaDispatches_=0;','globalHalfAlphaDispatches_=0;joinedGlobalMlpBlocks_=0;'),
 ('globalHalfAlphaDispatches_=gpu_->LastGlobalHalfAlphaDispatches();','globalHalfAlphaDispatches_=gpu_->LastGlobalHalfAlphaDispatches();joinedGlobalMlpBlocks_=gpu_->LastJoinedGlobalMlpBlocks();')])
edit('src/entry/metal_main.cpp',[('info+="; global_half_alpha";','info+="; global_half_alpha";\n#endif\n#ifdef GARNET_METAL_JOINED_GLOBAL_MLP\n        info+="; joined_global_mlp";')])
for name in ('xModel/qwen/metal_dense/decoder.py','xModel/qwen35/vl_0_8b/metal_decoder.py'):
 edit(name,[("self.global_half_alpha_enabled='global_half_alpha' in backend","self.global_half_alpha_enabled='global_half_alpha' in backend\n        self.joined_global_mlp_enabled='joined_global_mlp' in backend")])
edit('xModel/qwen35/vl_0_8b/metal_decoder.py',[("if self.global_half_alpha_enabled:self.prefill_profile['global_half_alpha_dispatches']=0","if self.global_half_alpha_enabled:self.prefill_profile['global_half_alpha_dispatches']=0\n        if self.joined_global_mlp_enabled:self.prefill_profile['joined_global_mlp_blocks']=0")])
for name in ('test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py'):
 edit(name,[("global_half_alpha_enabled='global_half_alpha' in garnet.backend_info(),","global_half_alpha_enabled='global_half_alpha' in garnet.backend_info(),\n            joined_global_mlp_enabled='joined_global_mlp' in garnet.backend_info(),")])
print('Added reset, successful-submit publication and actual prefill aggregation')
