#!/usr/bin/env python3
"""Summarize immutable request/Prometheus/process evidence from queue study."""
import json,re,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'.local/queue-capacity'
def metrics(s):
 d={}
 for line in s.splitlines():
  if not line or line.startswith('#'):continue
  key,value=line.rsplit(' ',1)
  try:d[key]=float(value)
  except ValueError:pass
 return d
def diff(a,b):
 a=metrics(a);b=metrics(b);return {k:v-a.get(k,0) for k,v in b.items()}
def timing(d,name,stage):
 key='voice_stage_'+name+'_seconds';suffix='{stage="'+stage+'"}'
 n=d.get(key+'_count'+suffix,0);total=d.get(key+'_sum'+suffix,0)
 if not n:return None
 buckets=[]
 for k,v in d.items():
  m=re.fullmatch(re.escape(key)+r'_bucket\{stage="'+stage+r'",le="([^"]+)"\}',k)
  if m:buckets.append((float(m[1]),v))
 return {'n':n,'mean_s':total/n,'p95_upper_s':next((b for b,c in sorted(buckets) if c>=.95*n),None),'over_2_05':n-next((v for b,v in buckets if b==2.05),0)}
def summarize(path):
 s=json.loads(path.read_text());case=path.parent
 a=s['cgroup_before'];b=s['cgroup_after'];elapsed=s['elapsed_s'];cpu=(int(b['cpu']['usage_usec'])-int(a['cpu']['usage_usec']))/1e6
 result={'planned_attempts':s.get('planned_attempts',s['users']*s['turns_each']),'stopped_on_failure':s.get('stopped_on_failure',False),'case':str(case.relative_to(OUT)),'users':s['users'],'mode':s['mode'],'completed':s['completed'],'attempts':s['attempts'],'errors':s['errors'],'elapsed_s':elapsed,'cpu_cores':cpu/elapsed,'cpu_s_per_completed':cpu/max(1,s['completed']),'peak_mib':b['memory_peak']/2**20,'throttled_s':(int(b['cpu']['throttled_usec'])-int(a['cpu']['throttled_usec']))/1e6,'client_schedule_lag_max_s':s['max_schedule_lag_s'],'send_lag':s['send_lag_max_s'],'latency':s['speech_end_to_audio_s'],'stt_latency':s['stt_after_finish_s'],'tts_latency':s['tts_first_audio_s']}
 before=json.loads((case/'workers-before.json').read_text());after=json.loads((case/'workers-after.json').read_text());native={}
 for stage,ports in [('stt',[p for p in before if int(p)<19100]),('tts',[p for p in before if int(p)>=19100])]:
  ds=[diff(before[p],after[p]) for p in ports];d={k:sum(x.get(k,0) for x in ds) for k in set().union(*ds)}
  native[stage]={n:timing(d,n,stage) for n in ('queue_wait','expired_queue_wait','inference','native_cpu')};native[stage]['rejections']={k:v for k,v in d.items() if 'admission_rejected' in k}
 result['stages']=native
 d=diff((case/'metrics-before.txt').read_text(),(case/'metrics-after.txt').read_text());result['gateway']={stage:{n:timing(d,n,stage) for n in ('queue_wait','inference','native_cpu','rpc','event_loop_delay')} for stage in ('vad','codec','gateway','stt','tts')}
 if (case/'samples.jsonl').exists():
  rows=[json.loads(l) for l in (case/'samples.jsonl').read_text().splitlines()];first,last=rows[0],rows[-1];start={p['pid']:p for p in first['processes']};stage_cpu={}
  for p in last['processes']:
   if p['pid'] in start:
    key=p['role']+('_native' if p['native'] else '_parent');stage_cpu[key]=stage_cpu.get(key,0)+p['cpu_s']-start[p['pid']]['cpu_s']
  result['process_cpu_s']=stage_cpu
  rolling=[]
  for i in range(5,len(rows)):
   dt=rows[i]['time']-rows[i-5]['time'];usage=(int(rows[i]['cpu']['usage_usec'])-int(rows[i-5]['cpu']['usage_usec']))/1e6/dt;rolling.append(usage)
  result['max_5s_cpu_cores']=max(rolling,default=0);result['median_5s_cpu_cores']=statistics.median(rolling) if rolling else 0;result['sample_peak_mib']=max(r['memory'] for r in rows)/2**20
 rows=[json.loads(line) for line in (case/'requests.jsonl').read_text().splitlines()]
 result['driver_cpu_s']=s.get('driver_cpu_s')
 result['all_turns_send_lag_max_s']=max((r.get('send_lag_max_s',0) for r in rows),default=0)
 result['overload_response_max_s']=max((r['overload_response_s'] for r in rows if 'overload_response_s' in r),default=None)
 native_errors=[r for r in rows if 'overload_response_s' in r and 'VAD queue full' not in r.get('error','')]
 result['native_overload_response_max_s']=max((r['overload_response_s'] for r in native_errors),default=None)
 result['overload_response_over_2_25']=sum(r.get('overload_response_s',0)>2.25 for r in rows)
 return result
if __name__=='__main__':
 results=[summarize(p) for p in sorted(OUT.glob('*/*/summary.json'))];(OUT/'analysis.json').write_text(json.dumps(results,indent=2)+'\n')
 for r in results:print(json.dumps({k:r[k] for k in ('case','completed','attempts','cpu_cores','latency','process_cpu_s','max_5s_cpu_cores') if k in r}))
