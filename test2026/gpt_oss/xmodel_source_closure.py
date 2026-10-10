"""Keep generated TP XLang stage packages closed over relative imports."""
import ast
from pathlib import Path

repo = Path(__file__).resolve().parents[2]
model_root = repo / 'xModel' / 'gpt_oss' / '120b'
pipeline = ast.parse((repo / 'tools' / 'gpt_oss' / 'pipeline.py').read_text(encoding='utf-8'))
copy_sets = []
for node in ast.walk(pipeline):
    if not isinstance(node, ast.For) or not isinstance(node.target, ast.Name) or node.target.id != 'name':
        continue
    values = node.iter.elts if isinstance(node.iter, (ast.Tuple, ast.List)) else ()
    names = {elt.value for elt in values if isinstance(elt, ast.Constant) and isinstance(elt.value, str)}
    if 'gpt_oss_llm.py' in names:
        copy_sets.append(names)
assert len(copy_sets) == 2, len(copy_sets)

def relative_dependencies(path):
    tree = ast.parse(path.read_text(encoding='utf-8'))
    dependencies = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 1:
            if node.module:
                dependencies.add(node.module + '.py')
            else:
                dependencies.update(alias.name + '.py' for alias in node.names)
    return dependencies

base = {'__init__.py', 'tensor_compat.py', 'fusion_policy.py', 'gpt_oss_llm.py', 'model.json'}
assert all(base <= names for names in copy_sets)
for names in copy_sets:
    assert relative_dependencies(model_root / 'gpt_oss_llm.py') <= names
assert 'fusion_policy.py' in relative_dependencies(model_root / 'gpt_oss_llm.py')
hybrid_source = (repo / 'tools' / 'gpt_oss' / 'pipeline.py').read_text(encoding='utf-8')
assert "shutil.copy2(root / 'gpt_oss_hybrid_llm.py', model_root / 'gpt_oss_hybrid_llm.py')" in hybrid_source
assert relative_dependencies(model_root / 'gpt_oss_hybrid_llm.py') <= base | {'gpt_oss_hybrid_llm.py'}
print('GPT-OSS generated TP XLang relative-import closure PASS')
