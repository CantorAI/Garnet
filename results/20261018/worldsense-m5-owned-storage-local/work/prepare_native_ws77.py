"""Freeze the original native proof plus independently checked ownership cases."""
from pathlib import Path
import ast
b=Path('/Users/shawn/Documents/Codex/2026-10-15/i');w=b/'work/worldsense-owned-storage-local';old=w.parent/'worldsense-minimal-selective-defaults'
t=(old/'native_controller.py').read_text().replace('worldsense-minimal-selective-defaults','worldsense-owned-storage-local').replace('len(manifest)==1721','len(manifest)==1722')
t=t.replace("roles=['off','primary','on','fault','compat56','compat56-fault']","roles=['off','primary','on','fault','compat56','compat56-fault','control','raw-borrow']")
t=t.replace("'read-only-consumer-prior-audit.json','source-transformations.json','immutable57-policy-guard.txt','build.py','build-plan.json'","'read-only-owner-prior-audit.json','source-transformations.json','build.py','build-plan.json','host-codegen-review.json'")
t=t.replace("['design_selective_ws74.py','setup_selective_ws74.py','prepare_native_ws74.py','verify_native_ws74.py']","['design_owned_storage_ws77.py','setup_owned_storage_ws77.py','prepare_native_ws77.py','verify_native_ws77.py','review_codegen_ws77.py']")
t=t.replace("assert len(manifest)==1722 and all(sha(s/k)==v for k,v in manifest.items())","assert len(manifest)==1722 and all(sha(s/k)==v for k,v in manifest.items())\nassert load(w/'host-codegen-review.json')['selected_local_ownership_removed_both_raw_and_packed']")
t=t.replace("for role in ['on','fault']:assert programs", "for role in ['on','fault','control']:assert programs")
anchor=" assert all(not programs[role]['core']for role in roles)";assert t.count(anchor)==1
t=t.replace(anchor," assert programs['raw-borrow']==programs['off']\n"+anchor)
t=t.replace('Native9tests','Native10tests').replace('all_six','all_eight').replace('ALL6ROLES9TESTS','ALL8ROLES10TESTS').replace('all_six','all_eight')
ast.parse(t);(w/'native_controller.py').write_text(t)
print('Eight-role original native controller prepared with ten tests per role; requires verified actual raw+packed host codegen before dispatch.')
