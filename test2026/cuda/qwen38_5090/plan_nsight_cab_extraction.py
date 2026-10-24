"""Plan read-only extraction of Windows profiler payload from the signed MSI."""
from pathlib import Path
import ctypes,json,re
from ctypes import wintypes as w
r=Path('D:/CantorAI/5090-study-20261023');m=ctypes.WinDLL('msi.dll');u=w.UINT
m.MsiOpenDatabaseW.argtypes=[w.LPCWSTR,w.LPCWSTR,ctypes.POINTER(u)];m.MsiDatabaseOpenViewW.argtypes=[u,w.LPCWSTR,ctypes.POINTER(u)];m.MsiViewExecute.argtypes=[u,u];m.MsiViewFetch.argtypes=[u,ctypes.POINTER(u)];m.MsiRecordGetStringW.argtypes=[u,u,w.LPWSTR,ctypes.POINTER(w.DWORD)];m.MsiCloseHandle.argtypes=[u]
def check(code):
 if code:raise RuntimeError('MSI read-only API status '+str(code))
db=u();check(m.MsiOpenDatabaseW(str(r/'tools/NsightSystems-2026.5.1.161.msi'),None,ctypes.byref(db)))
def query(sql,width):
 view=u();check(m.MsiDatabaseOpenViewW(db,sql,ctypes.byref(view)));check(m.MsiViewExecute(view,0));rows=[]
 while True:
  record=u();code=m.MsiViewFetch(view,ctypes.byref(record))
  if code==259:break
  check(code);row=[]
  for field in range(1,width+1):
   buf=ctypes.create_unicode_buffer(4096);n=w.DWORD(4096);check(m.MsiRecordGetStringW(record,field,buf,ctypes.byref(n)));row.append(buf.value)
  m.MsiCloseHandle(record);rows.append(row)
 m.MsiCloseHandle(view);return rows
directories={a:(b,c) for a,b,c in query('SELECT `Directory`, `Directory_Parent`, `DefaultDir` FROM `Directory`',3)}
components={a:b for a,b in query('SELECT `Component`, `Directory_` FROM `Component`',2)}
files=query('SELECT `File`, `Component_`, `FileName`, `FileSize`, `Sequence` FROM `File`',5)
media=[{'last_sequence':int(last),'cabinet':cab.lstrip('#')} for last,cab in query('SELECT `LastSequence`, `Cabinet` FROM `Media`',2)]
def relative(directory):
 if directory=='INSTALLDIR':return ''
 if directory not in directories:return None
 parent,default=directories[directory];base=relative(parent)
 if base is None:return None
 name=default.split(':',1)[0].split('|')[-1]
 if name=='.':return base
 assert name and '/' not in name and '\\' not in name and name!='..'
 return base+'/'+name if base else name
selected=[]
for key,component,name,size,sequence in files:
 directory=relative(components[component])
 if directory is None:continue
 top=directory.split('/')[0].lower()
 if top.startswith('target-') and top!='target-windows-x64':continue
 if top.startswith('host-') and top!='host-windows-x64':continue
 assert re.fullmatch(r'[A-Za-z0-9_.]+',key),key
 name=name.split('|')[-1];assert '/' not in name and '\\' not in name and name not in ('.','..')
 relative_path=directory+'/'+name if directory else name;cab=next(v['cabinet'] for v in sorted(media,key=lambda v:v['last_sequence']) if int(sequence)<=v['last_sequence'])
 selected.append({'key':key,'relative_path':relative_path,'bytes':int(size),'cabinet':cab})
m.MsiCloseHandle(db)
out=r/'results/nsight-2026-cab-plan';out.mkdir(exist_ok=False);(out/'plan.json').write_text(json.dumps({'files':selected,'media':media,'scope':'Read-only MSI file/directory mapping; Windows host/target plus shared files only; no executable or installer action.'},indent=2),encoding='utf8')
print(json.dumps({'files':len(selected),'bytes':sum(v['bytes'] for v in selected),'cabinets':media,'nsys':[v for v in selected if v['relative_path'].endswith('/nsys.exe')],'longest_relative_path':max(len(v['relative_path']) for v in selected)},indent=2))
