"""Persist B2 first/last trained layers at chunks16/32/64 with original bounds."""
from pathlib import Path
source=Path(__file__).with_name('test_qwen35_extended_prefill_reference.py').read_text()
source=source.replace('for batch in (1,2):','for batch in (2,):').replace('for chunk in ((40,80) if batch==1 else (40,)):','for chunk in (16,32,64):')
needle='for kind in (\'linear\',\'attention\'):';assert source.count(needle)==1
source=source.replace(needle,'''from contextlib import contextmanager
@contextmanager
def saved_fixture(kind,layer,batch,chunk):
    target=Path(sys.argv[5])/'fixtures'/(kind+'-l'+str(layer)+'-b'+str(batch)+'-c'+str(chunk))
    target.mkdir(parents=True,exist_ok=False)
    yield str(target)
'''+needle)
source=source.replace("with tempfile.TemporaryDirectory(prefix='metal-chunk-') as temp:",'with saved_fixture(kind,layer,batch,chunk) as temp:')
exec(compile(source,__file__,'exec'))
