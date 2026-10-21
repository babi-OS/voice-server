"""Local HPA up/down test using real isolated stage work; no production context."""
import argparse,asyncio,json,subprocess,time
from pathlib import Path
import aiohttp
from common import wait_ready_port
ROOT=Path(__file__).resolve().parents[1]; K=['kubectl','--kubeconfig',str(ROOT/'.local/kubeconfig'),'--context','k3d-voice-local','-n','voice-split-bench']
def snapshot(): return json.loads(subprocess.check_output(K+['get','hpa','-o','json'],text=True))
async def main(up_only=False):
    subprocess.run(K+['delete','hpa','voice-gateway','voice-stt','voice-tts','--ignore-not-found'],check=True,stdout=subprocess.DEVNULL)
    subprocess.run(K+['scale','deployment','--all','--replicas=1'],check=True,stdout=subprocess.DEVNULL)
    for role in ('gateway','stt','tts'):
        await asyncio.to_thread(subprocess.run,K+['rollout','status','deployment/voice-'+role,'--timeout=180s'],check=True,stdout=subprocess.DEVNULL)
    hpa=subprocess.check_output(['python3',str(ROOT/'local/hpa.py')],text=True)
    subprocess.run(K+['apply','-f','-'],input=hpa,text=True,check=True,stdout=subprocess.DEVNULL)
    out=ROOT/('.local/results/hpa-up-only' if up_only else '.local/results/hpa-cycle'); out.mkdir(parents=True,exist_ok=True); forwards=[]; rows=[]
    headers={'Authorization':'Bearer '+(ROOT/'.local/rpc-token').read_text().strip()}
    import wave
    with wave.open(str(ROOT/'tests/fixtures/vi_weather_two_clauses.wav')) as f: pcm=f.readframes(f.getnframes())*6
    try:
        for stage,port in [('stt',18771),('tts',18772)]:
            forwards.append(subprocess.Popen(K+['port-forward','service/voice-'+stage,f'{port}:8765','--address','127.0.0.1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
        await asyncio.sleep(2)
        stop=time.monotonic()+65; counts={s:{'completed':0,'overloaded':0,'other':0} for s in ('stt','tts')}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=125)) as session:
            for port,forward in zip((18771,18772),forwards): await wait_ready_port(session,port,forward)
            stop=time.monotonic()+65
            async def load(stage,port):
                while time.monotonic()<stop:
                    kwargs={'data':pcm} if stage=='stt' else {'json':{'text':'Hôm nay trời khá nóng. Bạn nên mang theo ô và uống đủ nước khi ra ngoài.','sample_rate':24000}}
                    try:
                        async with session.post(f'http://127.0.0.1:{port}/internal/v1/{stage}',headers=headers,**kwargs) as response:
                            await response.read(); counts[stage]['completed' if response.status==200 else 'overloaded' if response.status==429 else 'other']+=1
                    except aiohttp.ClientError: counts[stage]['other']+=1
                    await asyncio.sleep(.02)
            tasks=[asyncio.create_task(load(stage,port)) for stage,port in [('stt',18771),('tts',18772)] for _ in range(3)]
            while time.monotonic()<stop:
                rows.append({'timestamp':time.time(),'hpa':await asyncio.to_thread(snapshot)}); await asyncio.sleep(5)
            await asyncio.gather(*tasks)
        idle_started=time.time()
        for _ in range(0 if up_only else 80):
            sample={'timestamp':time.time(),'hpa':await asyncio.to_thread(snapshot)}; rows.append(sample)
            workers=[h for h in sample['hpa']['items'] if h['metadata']['name'] in ('voice-stt','voice-tts')]
            if len(workers)==2 and all(h.get('status',{}).get('currentReplicas')==1 and h.get('status',{}).get('desiredReplicas')==1 for h in workers): break
            await asyncio.sleep(5)
        else:
            if not up_only: raise RuntimeError('workers did not scale down within 400s idle')
        maxima={}
        for sample in rows:
            for h in sample['hpa']['items']:
                n=h['metadata']['name']; maxima[n]=max(maxima.get(n,0),h.get('status',{}).get('currentReplicas',0))
        assert maxima.get('voice-stt')==2 and maxima.get('voice-tts')==2
        report={'ok':True,'started_at_one_replica':True,'up_only':up_only,'stage_load':counts,'observed_maxima':maxima,'idle_to_one_replica_s':None if up_only else time.time()-idle_started,'note':'direct stage load through per-service port-forward tests HPA; gateway EndpointSlice distribution tested separately'}
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
    finally:
        (out/'snapshots.json').write_text(json.dumps(rows,indent=2)+'\n')
        for forward in forwards: forward.terminate(); await asyncio.to_thread(forward.wait)
        subprocess.run(K+['delete','hpa','--all'],check=True,stdout=subprocess.DEVNULL)
if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--up-only',action='store_true'); args=parser.parse_args()
    asyncio.run(main(args.up_only))
