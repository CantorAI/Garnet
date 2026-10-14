"""Add actual head dispatch counters, reset/publication and per-step logging."""
from pathlib import Path
root=Path(__file__).resolve().parents[2]
assert 'singleCoderHeadDispatches_' not in (root/'src/backends/metal/decoder_session.h').read_text(encoding='utf-8'),'Preparation already applied'
def edit(name,changes):
 p=root/name;s=p.read_text(encoding='utf-8')
 for before,after in changes:assert before in s,(name,before);s=s.replace(before,after)
 p.write_bytes(s.encode())
edit('src/backends/metal/metal_context.mm',[('impl_->constantHeadDispatches=0;','impl_->constantHeadDispatches=0;impl_->singleCoderHeadDispatches=0;')])
edit('src/backends/metal/decoder_session.h',[
 ('size_t constantHeadDispatches_=0;','size_t constantHeadDispatches_=0;\n    size_t singleCoderHeadDispatches_=0;'),
 ('size_t LastConstantHeadDispatches() const {return constantHeadDispatches_;}','size_t LastConstantHeadDispatches() const {return constantHeadDispatches_;}\n    size_t LastSingleCoderHeadDispatches() const {return singleCoderHeadDispatches_;}'),
 ('APISET().AddProp("constant_head_dispatches",&MetalDecoderSession::LastConstantHeadDispatches);','APISET().AddProp("constant_head_dispatches",&MetalDecoderSession::LastConstantHeadDispatches);\n        APISET().AddProp("single_coder_head_dispatches",&MetalDecoderSession::LastSingleCoderHeadDispatches);')])
edit('src/backends/metal/decoder_session.cpp',[
 ('constantHeadDispatches_=0;','constantHeadDispatches_=0;singleCoderHeadDispatches_=0;'),
 ('constantHeadDispatches_=gpu_->LastConstantHeadDispatches();','constantHeadDispatches_=gpu_->LastConstantHeadDispatches();singleCoderHeadDispatches_=gpu_->LastSingleCoderHeadDispatches();')])
edit('src/entry/metal_main.cpp',[('info+="; constant_bf16_head";','info+="; constant_bf16_head";\n#endif\n#ifdef GARNET_METAL_SINGLE_CODER_HEAD\n        info+="; single_coder_head";')])
edit('test2026/metal/run_matched_garnet.py',[
 ('initialization=time.monotonic()-started', "single_coder_head_enabled='single_coder_head' in garnet.backend_info()\n    initialization=time.monotonic()-started"),
 ('prediction=decoder.prefill_sequence(hidden,positions);ttft=time.monotonic()-begin', 'prediction=decoder.prefill_sequence(hidden,positions);ttft=time.monotonic()-begin\n        single_head_counts=[decoder.session.single_coder_head_dispatches if single_coder_head_enabled else 0]'),
 ('durations.append(time.monotonic()-step);native_ms.append(decoder.last_session_ms)', 'durations.append(time.monotonic()-step);native_ms.append(decoder.last_session_ms)\n            single_head_counts.append(decoder.session.single_coder_head_dispatches if single_coder_head_enabled else 0)'),
 ("joined_global_mlp_enabled='joined_global_mlp' in garnet.backend_info(),", "joined_global_mlp_enabled='joined_global_mlp' in garnet.backend_info(),\n            single_coder_head_enabled=single_coder_head_enabled,single_coder_head_counts=single_head_counts,")])
# VL has a different vocabulary shape. Record actual zero usage without a
# property lookup in its decode timing loop; its final session is still checked.
edit('test2026/metal/frame_worker_garnet.py',[("joined_global_mlp_enabled='joined_global_mlp' in garnet.backend_info(),", "joined_global_mlp_enabled='joined_global_mlp' in garnet.backend_info(),\n            single_coder_head_enabled='single_coder_head' in garnet.backend_info(),\n            single_coder_head_counts=[decoder.session.single_coder_head_dispatches if 'single_coder_head' in garnet.backend_info() else 0],")])
print('Added actual head counters and fresh decode-step telemetry')
