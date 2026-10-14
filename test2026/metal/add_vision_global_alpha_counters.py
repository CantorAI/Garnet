"""Reset context/graph counters and record actual vision dispatches per image."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
assert 'impl_->visionGlobalAlphaDispatches=0;' not in (root/'src/backends/metal/metal_context.mm').read_text(encoding='utf-8'),'Preparation already applied'
def edit(name,changes):
 p=root/name;s=p.read_text(encoding='utf-8')
 for before,after in changes:assert before in s,(name,before);s=s.replace(before,after)
 p.write_bytes(s.encode())
edit('src/backends/metal/metal_context.mm',[('impl_->globalHalfAlphaDispatches=0;','impl_->globalHalfAlphaDispatches=0;impl_->visionGlobalAlphaDispatches=0;')])
edit('src/backends/metal/compiled_graph.cpp',[('++generation_;ready_=false;','visionGlobalAlphaDispatches_=0;++generation_;ready_=false;')])
edit('src/entry/metal_main.cpp',[('info+="; global_half_alpha";','info+="; global_half_alpha";\n#endif\n#ifdef GARNET_METAL_VISION_GLOBAL_ALPHA\n        info+="; vision_global_alpha";')])
edit('test2026/metal/run_matched_garnet.py',[("single_coder_head_enabled=single_coder_head_enabled,single_coder_head_counts=single_head_counts,","single_coder_head_enabled=single_coder_head_enabled,single_coder_head_counts=single_head_counts,\n            vision_global_alpha_enabled='vision_global_alpha' in garnet.backend_info(),vision_global_alpha_dispatches=0,")])
edit('test2026/metal/frame_worker_garnet.py',[("single_coder_head_enabled='single_coder_head' in garnet.backend_info(),","single_coder_head_enabled='single_coder_head' in garnet.backend_info(),\n            vision_global_alpha_enabled='vision_global_alpha' in garnet.backend_info(),\n            vision_global_alpha_dispatches=vision.graph.vision_global_alpha_dispatches if 'vision_global_alpha' in garnet.backend_info() else 0,")])
print('Added reset and successful-run vision use telemetry')
