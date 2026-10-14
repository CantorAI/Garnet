"""Pin the five route-only changes against the archived all-head candidate."""
from pathlib import Path
source=Path(__file__).with_name('prepare_fused_head_argmax_sources.py').read_text()
source=source.replace('fused-head-argmax-staging','decode-only-head-staging').replace('4d1f807c:','88487eb9:')
start=source.index('paths=[');end=source.index('original={}',start)
source=source[:start]+"paths=['cmake/GarnetMetal.cmake','src/backends/metal/decoder_session.cpp','src/entry/metal_main.cpp','test2026/metal/run_matched_garnet.py','test2026/metal/frame_worker_garnet.py']\nnew=set();"+source[end:]
source=source.replace('install_fused_head_argmax_sources.py','install_decode_only_head_sources.py')
exec(compile(source,__file__,'exec'))
