from pathlib import Path
import ctypes,json
from ctypes import wintypes as w
m=ctypes.WinDLL('msi.dll');u=w.UINT
m.MsiOpenDatabaseW.argtypes=[w.LPCWSTR,w.LPCWSTR,ctypes.POINTER(u)];m.MsiDatabaseOpenViewW.argtypes=[u,w.LPCWSTR,ctypes.POINTER(u)];m.MsiViewExecute.argtypes=[u,u];m.MsiViewFetch.argtypes=[u,ctypes.POINTER(u)];m.MsiRecordGetStringW.argtypes=[u,u,w.LPWSTR,ctypes.POINTER(w.DWORD)];m.MsiCloseHandle.argtypes=[u]
def check(code):
 if code:raise RuntimeError('MSI read-only API status '+str(code))
db=u();check(m.MsiOpenDatabaseW('D:/CantorAI/5090-study-20261023/tools/NsightSystems-2026.5.1.161.msi',None,ctypes.byref(db)));view=u();check(m.MsiDatabaseOpenViewW(db,'SELECT `Directory`, `Directory_Parent`, `DefaultDir` FROM `Directory`',ctypes.byref(view)));check(m.MsiViewExecute(view,0));rows=[]
while True:
 record=u();code=m.MsiViewFetch(view,ctypes.byref(record))
 if code==259:break
 check(code);values=[]
 for field in (1,2,3):
  buf=ctypes.create_unicode_buffer(4096);n=w.DWORD(4096);check(m.MsiRecordGetStringW(record,field,buf,ctypes.byref(n)));values.append(buf.value)
 m.MsiCloseHandle(record);rows.append(values)
m.MsiCloseHandle(view);m.MsiCloseHandle(db)
r=Path('D:/CantorAI/5090-study-20261023/results/nsight-2026-tool-setup-short-fix');(r/'msi-directory-table.json').write_text(json.dumps(rows,indent=2),encoding='utf8')
print(json.dumps([v for v in rows if v[1]=='TARGETDIR' or any(k in v[2].lower() for k in ('nsight systems','nvidia corporation','target-windows-x64'))],indent=2))
