"""Real CPU interpreter/package regression; synthetic tokenizer, no GPU proof."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import venv

repo=Path(__file__).resolve().parents[2]
validator=repo/'tools/gpt_oss/validate_batch_results.py'
with tempfile.TemporaryDirectory() as temporary:
    root=Path(temporary);environment=root/'venv'
    venv.EnvBuilder(with_pip=False,symlinks=os.name=='posix').create(environment)
    python=environment/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    clean=os.environ.copy();clean.pop('PYTHONPATH',None);clean.pop('PYTHONHOME',None)
    package=Path(subprocess.check_output([str(python),'-c',
        "import sysconfig;print(sysconfig.get_path('purelib'))"],env=clean,text=True).strip())/'transformers'
    package.mkdir()
    # Only this isolated venv has the deliberately tiny dependency. The actual
    # production validator must load it through its preserved launcher path.
    (package/'__init__.py').write_text('''import json
from pathlib import Path
class AutoTokenizer:
    @staticmethod
    def from_pretrained(model, local_files_only):
        assert local_files_only is True
        assert json.loads((Path(model)/'tokenizer.json').read_text()) == {'cpu_fixture': True}
        print('ISOLATED_VENV_CPU_DEPENDENCY')
        return object()
''')
    model=root/'model';model.mkdir();(model/'tokenizer.json').write_text(json.dumps({'cpu_fixture':True}))
    def run(executable,folder):
        return subprocess.run([str(executable),str(validator),'--preflight',str(folder)],
            env=clean,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    passed=run(python.absolute(),model)
    assert passed.returncode==0 and 'ISOLATED_VENV_CPU_DEPENDENCY' in passed.stdout,passed.stdout
    if os.name=='posix':
        assert python.is_symlink() and python.resolve()!=python.absolute()
        wrong=run(python.resolve(),model)
        assert wrong.returncode!=0 and 'ModuleNotFoundError' in wrong.stdout,wrong.stdout
    missing=run(python,root/'missing-tokenizer')
    assert missing.returncode!=0 and 'FileNotFoundError' in missing.stdout,missing.stdout
    (model/'tokenizer.json').write_text('{}')
    invalid=run(python,model)
    assert invalid.returncode!=0 and 'AssertionError' in invalid.stdout,invalid.stdout
    # Remove the fixture dependency using its exact verified temporary path.
    dependency=package/'__init__.py';assert dependency.is_relative_to(root)
    dependency.unlink()
    # The empty namespace package cannot supply AutoTokenizer; stale pycache
    # does not recreate __init__.py and must not turn this into a pass.
    missing_dependency=run(python,model)
    assert missing_dependency.returncode!=0 and 'ImportError' in missing_dependency.stdout,missing_dependency.stdout
print('Actual validator CPU venv/local-tokenizer preflight and dependency/tokenizer failures PASS; symlink regression exercised on POSIX; no real model/GPU proof')
