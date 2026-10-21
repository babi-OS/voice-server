"""Real inference queue deadline and saturation on a fresh disposable TTS Pod."""
import asyncio,json,subprocess,time
from pathlib import Path
import aiohttp
from common import wait_ready_port
ROOT=Path(__file__).resolve().parents[1]; K=['kubectl','--kubeconfig',str(ROOT/'.local/kubeconfig'),'--context','k3d-voice-local','-n','voice-split-bench']; NAME='voice-queue-probe'
def k(*args):return subprocess.check_output(K+list(args),text=True)
async def main():
    template=json.loads(k('get','deployment','voice-tts','-o','json'))['spec']['template']
    pod={'apiVersion':'v1','kind':'Pod','metadata':{'name':NAME,'namespace':'voice-split-bench','labels':{'app':NAME}},'spec':template['spec']}
    subprocess.run(K+['apply','-f','-'],input=json.dumps(pod),text=True,check=True,stdout=subprocess.DEVNULL)
    forward=None; tasks=[]; report={}
    try:
        await asyncio.to_thread(k,'wait','pod/'+NAME,'--for=condition=Ready','--timeout=180s')
        forward=subprocess.Popen(K+['port-forward','pod/'+NAME,'18777:8765','--address','127.0.0.1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        headers={'Authorization':'Bearer '+(ROOT/'.local/rpc-token').read_text().strip(),'X-Queue-Budget-Ms':'2000'}
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=125)) as session:
            await wait_ready_port(session,18777,forward)
            async def job(text):
                started=time.monotonic()
                async with session.post('http://127.0.0.1:18777/internal/v1/tts',headers=headers,json={'text':text,'sample_rate':24000}) as response:
                    body=await response.read(); return {'status':response.status,'elapsed_s':time.monotonic()-started,'bytes':len(body)}
            async def admitted(n):
                for _ in range(100):
                    async with session.get('http://127.0.0.1:18777/capacity',headers=headers) as response: data=await response.json()
                    if data['admitted']==n:return
                    await asyncio.sleep(.01)
                raise RuntimeError(f'expected {n} admitted jobs')
            primary=asyncio.create_task(job(('Hôm nay trời khá nóng. Bạn nên mang theo ô và uống đủ nước khi ra ngoài. '*22)[:1500])); tasks.append(primary)
            await admitted(1)
            queued=asyncio.create_task(job('Xin chào.')); tasks.append(queued); await admitted(2)
            full=await job('Xin chào.'); expired=await queued; finished=await primary
            assert full['status']==429 and full['elapsed_s']<.25
            assert expired['status']==429 and 1.9<=expired['elapsed_s']<=2.05,expired
            assert finished['status']==200 and finished['bytes']>10000
            report={'ok':True,'full_queue':full,'actual_queue_deadline':expired,'accepted_native_job':finished}
    finally:
        for task in tasks:
            if not task.done():task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        if forward:forward.terminate();await asyncio.to_thread(forward.wait)
        subprocess.run(K+['delete','pod',NAME,'--wait=false'],check=True,stdout=subprocess.DEVNULL)
    out=ROOT/'.local/results/queue-deadline';out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':asyncio.run(main())
