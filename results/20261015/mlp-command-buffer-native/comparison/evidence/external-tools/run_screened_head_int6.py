"""Same exact captured-query greedy checks with a packed derived INT6 index."""
from pathlib import Path
source=Path(__file__).with_name('run_screened_head.py').read_text()
source=source.replace('from make_screened_head_harness import generate','from make_screened_head_int6 import generate_harness as generate,generate_shader')
source=source.replace("s/'make_screened_head_harness.py'","s/'run_screened_head.py',s/'make_screened_head_int6.py',s/'prepare_screened_head_int6.py',s/'make_screened_head_harness.py'")
source=source.replace("+raw+argmax+(s/'screened_head.metal').read_text()","+raw+argmax+generate_shader((s/'screened_head.metal').read_text())")
source=source.replace("str(s/'prepare_screened_head.py')","str(s/'prepare_screened_head_int6.py')").replace('Derived INT8 filter index','Derived INT6 filter index')
exec(compile(source,__file__,'exec'))
