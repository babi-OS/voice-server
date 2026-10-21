"""Read native worker process CPU counters from /proc (no ps dependency)."""
import json,os,time
from pathlib import Path
rows=[]
for p in Path('/proc').iterdir():
    if not p.name.isdigit() or int(p.name)==os.getpid(): continue
    try:
        cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
        if 'multiprocessing.spawn' not in cmd: continue
        fields=(p/'stat').read_text().rsplit(')',1)[1].split()
        rows.append({'pid':int(p.name),'cpu_s':(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK')})
    except (OSError,ValueError): pass
print(json.dumps({'timestamp':time.time(),'workers':rows,'cpu_s':sum(r['cpu_s'] for r in rows)}))
