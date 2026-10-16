"""Include asynchronous native keys and graph execution in one candidate clock."""
from pathlib import Path

wrapper=Path(__file__).with_name('run_hybrid_combined_graph_prefill.py').read_text();ns={'__file__':__file__}
exec(compile(wrapper[:wrapper.rindex('exec(compile(source')],__file__,'exec'),ns);source=ns['source']
source=source.replace('make_hybrid_combined_graph_prefill.py','make_async_key_graph_prefill.py')
source=source.replace('hybrid_combined_graph_prefill_gpu','async_key_graph_prefill_gpu')
needle='paths=[Path(__file__),';assert source.count(needle)==1
source=source.replace(needle,needle+"s/'make_hybrid_combined_graph_prefill.py',s/'run_hybrid_combined_graph_prefill.py',")
source=source.replace('Candidate wall clocks include the native key submission and synchronous graph call.', 'Candidate wall clocks include native key encoding/Flush, graph execution and final native SubmitAndWait. No CPU wait occurs between the independent key and graph submissions; hardware overlap is not measured or assumed.')
exec(compile(source,__file__,'exec'))
