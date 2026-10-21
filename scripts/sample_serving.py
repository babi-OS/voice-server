"""Container-local bounded sampling, kept separate from request timings."""
import json,os,time,urllib.request
from pathlib import Path
out=Path('/tmp/voice-samples.jsonl'); stop=Path('/tmp/voice-sample-stop');stop.unlink(missing_ok=True)
with out.open('w') as f:
 for _ in range(1800):
  c=Path('/sys/fs/cgroup');row={'time':time.time(),'cpu':dict(line.split() for line in (c/'cpu.stat').read_text().splitlines()),'memory':int((c/'memory.current').read_text()),'processes':[]}
  for p in Path('/proc').iterdir():
   if not p.name.isdigit() or int(p.name)==os.getpid():continue
   try:
    env=dict(x.split(b'=',1) for x in (p/'environ').read_bytes().split(b'\0') if b'=' in x);role=env.get(b'VOICE_ROLE',b'').decode()
    if not role:continue
    cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode();fields=(p/'stat').read_text().rsplit(')',1)[1].split()
    row['processes'].append({'pid':int(p.name),'role':role,'native':'multiprocessing.spawn' in cmd,'cpu_s':(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),'rss':int(fields[21])*os.sysconf('SC_PAGE_SIZE')})
   except (OSError,ValueError):pass
  f.write(json.dumps(row)+'\n');f.flush()
  if stop.exists():break
  time.sleep(1)
