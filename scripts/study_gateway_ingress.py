#!/usr/bin/env python3
"""Local real-model queue/capacity study: all roles share 4 CPU/6 GiB."""
import argparse,json,os,secrets,subprocess,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]; NAME='voice-queue-study'; OUT=ROOT/'.local/queue-capacity'; PORTS=(19001,19002,19101,19102)
def run(args,**kw):return subprocess.run(args,check=True,**kw)
def capture():
    code="import urllib.request,json; print(json.dumps({str(p):urllib.request.urlopen('http://127.0.0.1:'+str(p)+'/metrics').read().decode() for p in "+repr(PORTS)+"}))"
    return json.loads(subprocess.check_output(['docker','exec',NAME,'/app/.venv/bin/python','-c',code],text=True))
def main():
    p=argparse.ArgumentParser();p.add_argument('--admission',type=int,required=True);p.add_argument('--asr-threads',type=int,default=2);p.add_argument('--tts-threads',type=int,default=2);p.add_argument('--tts-workers',type=int,default=2,choices=[1,2,3,4]);p.add_argument('--stt-workers',type=int,default=2,choices=[1,2]);p.add_argument('--image',default='voice-server:single-pass-ingress-release');p.add_argument('--users',type=int,nargs='+',required=True);p.add_argument('--mode',choices=['burst','staggered'],default='staggered');p.add_argument('--turns',type=int,default=4);p.add_argument('--label',required=True);p.add_argument('--tts-batch-size',type=int,choices=[1,2,4],default=1);p.add_argument('--asr-batch-size',type=int,choices=[1,2,4],default=1);p.add_argument('--batch-wait-ms',type=int,choices=range(1,11),default=2);p.add_argument('--tts-model-dir',type=Path);a=p.parse_args()
    if not 1<=a.admission<=256 or not 1<=a.asr_threads<=32 or not 1<=a.tts_threads<=32 or not 1<=a.turns<=100 or any(not 1<=n<=512 for n in a.users):p.error('bounded study: admission 1..256, users 1..512, threads 1..32, turns 1..100')
    if a.label in ('.','..') or Path(a.label).name!=a.label:p.error('label must be a single directory name')
    global PORTS
    PORTS=tuple(19000+i for i in range(1,a.stt_workers+1))+tuple(19100+i for i in range(1,a.tts_workers+1))
    out=OUT/a.label;out.mkdir(parents=True,exist_ok=False)
    token=secrets.token_urlsafe(24); image=a.image
    if subprocess.run(['docker','inspect',NAME],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0:raise RuntimeError('study container already exists')
    env={'HF_HOME':'/models/huggingface','HF_HUB_OFFLINE':'1','MATCHA_REVISION':'4f741be47b95f6ee1b23e15ba34f12c302fa64b7','VAD_REVISION':'e71cae966052b992a7eca6b17738916ce0eca4ec','VOICE_ORT_SPINNING':'0','MATCHED_MODE':'split','VOICE_RPC_TOKEN':token,'VOICE_ASR_THREADS':str(a.asr_threads),'VOICE_TTS_THREADS':str(a.tts_threads),'MATCHED_TTS_WORKERS':str(a.tts_workers),'MATCHED_STT_WORKERS':str(a.stt_workers),'VOICE_STT_ADMISSION':str(a.admission),'VOICE_TTS_ADMISSION':str(a.admission),'VOICE_GATEWAY_MAX_MICROPHONES':'256','VOICE_GATEWAY_MAX_CONNECTIONS':'512','OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}
    env.update(VOICE_TTS_BATCH_SIZE=str(a.tts_batch_size),VOICE_ASR_BATCH_SIZE=str(a.asr_batch_size),VOICE_TTS_BATCH_WAIT_MS=str(a.batch_wait_ms),VOICE_ASR_BATCH_WAIT_MS=str(a.batch_wait_ms))
    if a.tts_model_dir: env['MATCHA_CPU_DIR']='/batch-model'
    metadata=(vars(a)|{'tts_model_dir':str(a.tts_model_dir) if a.tts_model_dir else None})|{'image_id':subprocess.check_output(['docker','image','inspect',image,'--format','{{.Id}}'],text=True).strip(),'limits':{'cpu':4,'memory_gib':6,'cpuset':'0-5'},'workers_per_stage':{'stt':a.stt_workers,'tts':a.tts_workers},'tts_threads':a.tts_threads,'queue_budget_s':2,'max_microphones':256}
    (out/'provenance.json').write_text(json.dumps(metadata,indent=2)+'\n')
    args=['docker','run','-d','--name',NAME,'--cpus','4','--cpuset-cpus','0-5','--memory','6g','--memory-swap','6g','-p','127.0.0.1:18766:8765','-v','voice-server_voice-models:/models:ro','-v',str(ROOT)+':/source:ro','-v',str(ROOT/'scripts')+':/benchmark:ro']
    if a.tts_model_dir: args+=['-v',str(a.tts_model_dir.resolve())+':/batch-model:ro']
    for k,v in env.items():args+=['-e',k+'='+v]
    run(args+['--entrypoint','/app/.venv/bin/python',image,'/benchmark/matched_service.py'],stdout=subprocess.DEVNULL)
    try:
        for _ in range(180):
            try:urllib.request.urlopen('http://127.0.0.1:18766/readyz',timeout=1);break
            except OSError:time.sleep(1)
        else:raise RuntimeError('not ready')
        warm='''import asyncio,aiohttp,wave,json,os,time
async def main():
 with wave.open('/source/tests/fixtures/vi_weather_two_clauses.wav') as f:pcm=f.readframes(f.getnframes())
 async with aiohttp.ClientSession(headers={'Authorization':'Bearer '+os.environ['VOICE_RPC_TOKEN']}) as s:
  async def backend(p):
   rows=[]
   for i in range(8):
    start=time.monotonic(); kw={'data':pcm} if p<19100 else {'json':{'text':'Hôm nay trời khá nóng. Bạn nên mang theo ô và uống đủ nước khi ra ngoài.','sample_rate':24000}}
    async with s.post('http://127.0.0.1:'+str(p)+'/internal/v1/'+('stt' if p<19100 else 'tts'),**kw) as r:
     body=await r.read();assert r.status==200,(r.status,body);rows.append({'wall_s':time.monotonic()-start,'bytes':len(body),'queue_ms':r.headers.get('X-Queue-Wait-Ms'),'inference_ms':r.headers.get('X-Inference-Ms')})
   return rows
  print(json.dumps(dict(zip(PORTS,await asyncio.gather(*(backend(p) for p in PORTS))))))
asyncio.run(main())'''
        warm=warm.replace('PORTS',repr(PORTS))
        with (out/'warmup.json').open('w') as f:run(['docker','exec',NAME,'/app/.venv/bin/python','-c',warm],stdout=f)
        time.sleep(2)
        for users in a.users:
            case=out/str(users);case.mkdir()
            (case/'workers-before.json').write_text(json.dumps(capture(),indent=2)+'\n')
            run(['docker','exec','-d',NAME,'/app/.venv/bin/python','/benchmark/sample_serving.py'])
            with (case/'driver.log').open('w') as f:
                run(['taskset','-c','19-20',sys.executable,str(ROOT/'scripts/bench_ingress.py'),'--url','ws://127.0.0.1:18766/v2/voice','--users',str(users),'--turns',str(a.turns),'--mode',a.mode,'--container',NAME,'--output',str(case)]+(['--stop-on-failure'] if a.turns>=40 else []),stdout=f,stderr=f)
            run(['docker','exec',NAME,'touch','/tmp/voice-sample-stop'])
            time.sleep(1.1)
            with (case/'samples.jsonl').open('w') as f:run(['docker','exec',NAME,'cat','/tmp/voice-samples.jsonl'],stdout=f)
            (case/'workers-after.json').write_text(json.dumps(capture(),indent=2)+'\n')
            s=json.loads((case/'summary.json').read_text()); c0=s['cgroup_before']['cpu'];c1=s['cgroup_after']['cpu'];cpu=(int(c1['usage_usec'])-int(c0['usage_usec']))/1e6/s['elapsed_s']
            print(json.dumps({'case':str(case.relative_to(OUT)),'completed':s['completed'],'attempts':s['attempts'],'p95':(s['speech_end_to_audio_s'] or {}).get('p95'),'cpu_cores':cpu,'errors':s['errors'],'client_lag':s['max_schedule_lag_s']}),flush=True)
            time.sleep(2)
    finally:
        with (out/'service.log').open('w') as f:subprocess.run(['docker','logs',NAME],stdout=f,stderr=f)
        subprocess.run(['docker','stop','-t','150',NAME],stdout=subprocess.DEVNULL);subprocess.run(['docker','rm',NAME],stdout=subprocess.DEVNULL)
if __name__=='__main__':main()
