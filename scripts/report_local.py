#!/usr/bin/env python3
"""Summarize saved local benchmark results; incomplete cases are never successes."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--results',default=str(Path(__file__).resolve().parents[1]/'.local/results')); p.add_argument('--output',required=True); a=p.parse_args()
rows=[]
for path in sorted(Path(a.results).rglob('summary.json')):
    data=json.loads(path.read_text())
    if not isinstance(data,dict) or 'attempts' not in data: continue
    latency=data.get('speech_end_to_audio_s')
    rows.append({'case':str(path.parent.relative_to(Path(a.results))),'users':data['users'],'mode':data['mode'],'completed':data['completed'],'attempts':data['attempts'],'p95_s':latency.get('p95') if latency else None,'elapsed_s':data['elapsed_s'],'max_client_lag_s':data['max_schedule_lag_s'],'passes_zero_failure_5s':data['failed']==0 and bool(latency) and latency['p95']<=5,'cgroup':data.get('cgroup_after')})
Path(a.output).write_text(json.dumps(rows,indent=2)+'\n')
for row in rows: print(row['case'],f"{row['completed']}/{row['attempts']}",'p95',row['p95_s'])
