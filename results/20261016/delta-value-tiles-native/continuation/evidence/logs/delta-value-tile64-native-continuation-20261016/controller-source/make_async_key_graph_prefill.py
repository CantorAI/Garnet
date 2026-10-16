"""Submit independent native keys asynchronously; preserve graph and key math."""
from pathlib import Path
import runpy,sys

runpy.run_path(str(Path(__file__).with_name('make_hybrid_combined_graph_prefill.py')),run_name='__main__')
path=Path(sys.argv[1]);source=path.read_text()
needle='native.SubmitAndWait();}int index=policy-1;';assert source.count(needle)==1
source=source.replace(needle,'native.Flush();}int index=policy-1;')
needle='resultsDictionary:results[index]];}else{native.Begin();';assert source.count(needle)==1
source=source.replace(needle,'resultsDictionary:results[index]];if(nativeKeys){native.SubmitAndWait();need(native.LastPartialSubmissions()==1,"Native key flush count mismatch");}}else{native.Begin();')
needle='@"native_key_bit_exact":@YES,';assert source.count(needle)==1
source=source.replace(needle,needle+'@"native_key_async_flushes":@(nativeKeys?native.LastPartialSubmissions():0),')
path.write_bytes(source.encode())
