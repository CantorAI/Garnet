"""Prospective text-only repeated native screen; full matched-vLLM target stays unmet."""
from pathlib import Path
p=Path(__file__).with_name('analyze_vector_vision_native_results.py');source=p.read_text()
source=source.replace("p['model']=='Qwen3.5-0.8B'","p['model']=='Qwen3-0.6B'").replace("p['ttft_fraction']<=.98","p['ttft_fraction']<=.97")
source=source.replace("assert audit['full_output_requests']==72 and audit['actual_vector_vision_routes'] and audit['regenerated_native_comparators']==6","assert audit['full_output_requests']==72 and audit['actual_prepared_qkv_routes'] and audit['complete_qkv_route_requests']==72 and audit['regenerated_native_comparators']==6")
source=source.replace('c186930919845dc172395f7a26564ce94a2a1fe2ebdb755363774a3fd54e29c1','d52c37e09edf7afb61889037863b1cb338f9ac0b6d98ea62043c3a016af5b55e')
source=source.replace('>=2% warm TTFT reduction BOTH VL batches','>=3% warm TTFT reduction BOTH text batches')
exec(compile(source,__file__,'exec'))
