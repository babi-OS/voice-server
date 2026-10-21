"""Validate real maximum-size internal jobs inside their existing Pod limits."""
import asyncio,json,subprocess,time,wave
from pathlib import Path
import aiohttp
from common import wait_ready_port
ROOT=Path(__file__).resolve().parents[1]; K=['kubectl','--kubeconfig',str(ROOT/'.local/kubeconfig'),'--context','k3d-voice-local','-n','voice-split-bench']
async def main():
    rows=[]; headers={'Authorization':'Bearer '+(ROOT/'.local/rpc-token').read_text().strip()}
    with wave.open(str(ROOT/'tests/fixtures/vi_weather_two_clauses.wav')) as f: fixture=f.readframes(f.getnframes())
    pcm=(fixture*7)[:960000]
    text=('Hôm nay trời khá nóng. Bạn nên mang theo ô và uống đủ nước khi ra ngoài. '*22)[:1500]
    for stage,body in [('stt',{'data':pcm}),('tts',{'json':{'text':text,'sample_rate':24000}})]:
        forward=subprocess.Popen(K+['port-forward','service/voice-'+stage,'18776:8765','--address','127.0.0.1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        try:
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=125)) as session:
                await wait_ready_port(session,18776,forward); started=time.monotonic()
                async with session.post('http://127.0.0.1:18776/internal/v1/'+stage,headers=headers,**body) as response:
                    payload=await response.read(); assert response.status==200,(stage,response.status,payload[:100])
                    rows.append({'stage':stage,'ok':True,'input_size':len(pcm) if stage=='stt' else len(text),'input_unit':'bytes' if stage=='stt' else 'characters','response_bytes':len(payload),'elapsed_s':time.monotonic()-started,'replica':response.headers.get('X-Voice-Replica')})
                invalid={'data':pcm+b'xx'} if stage=='stt' else {'json':{'text':text+'x','sample_rate':24000}}
                async with session.post('http://127.0.0.1:18776/internal/v1/'+stage,headers=headers,**invalid) as response:
                    assert response.status in (400,413); rows[-1]['over_limit_rejected_status']=response.status
        finally: forward.terminate(); await asyncio.to_thread(forward.wait)
    out=ROOT/'.local/results/maximum-input';out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__': asyncio.run(main())
