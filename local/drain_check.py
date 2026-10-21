"""Real native TTS survives a local Deployment scale-down from two Pods to one."""
import asyncio,json,subprocess,time
from pathlib import Path
import aiohttp
from common import wait_ready_port
ROOT=Path(__file__).resolve().parents[1]; K=['kubectl','--kubeconfig',str(ROOT/'.local/kubeconfig'),'--context','k3d-voice-local','-n','voice-split-bench']
async def main():
    pods=json.loads(subprocess.check_output(K+['get','pods','-l','app=voice-tts','-o','json'],text=True))['items']
    pods=[p for p in pods if not p['metadata'].get('deletionTimestamp')]
    assert len(pods)==2
    forwards=[]; tasks=[]; report={}; headers={'Authorization':'Bearer '+(ROOT/'.local/rpc-token').read_text().strip()}
    try:
        for i,pod in enumerate(pods):
            forwards.append(subprocess.Popen(K+['port-forward','pod/'+pod['metadata']['name'],f'{18773+i}:8765','--address','127.0.0.1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL))
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=125)) as session:
            for i,forward in enumerate(forwards): await wait_ready_port(session,18773+i,forward)
            async def job(port):
                async with session.post(f'http://127.0.0.1:{port}/internal/v1/tts',headers=headers,json={'text':'Hôm nay trời khá nóng. Bạn nên mang theo ô và uống đủ nước khi ra ngoài. '*14,'sample_rate':24000}) as response:
                    pcm=await response.read(); return {'status':response.status,'pcm_bytes':len(pcm),'queue_ms':response.headers.get('X-Queue-Wait-Ms'),'inference_ms':response.headers.get('X-Inference-Ms')}
            tasks=[asyncio.create_task(job(18773+i)) for i in range(2)]
            for _ in range(100):
                caps=[]
                for i in range(2):
                    async with session.get(f'http://127.0.0.1:{18773+i}/capacity',headers=headers) as response: caps.append(await response.json())
                if all(c['running']==1 for c in caps): break
                await asyncio.sleep(.05)
            assert all(c['running']==1 for c in caps),'did not observe both native jobs running'
            started=time.monotonic()
            await asyncio.to_thread(subprocess.run,K+['scale','deployment/voice-tts','--replicas=1'],check=True,stdout=subprocess.DEVNULL)
            results=await asyncio.gather(*tasks)
            assert all(r['status']==200 and r['pcm_bytes']>10000 for r in results),results
            report={'test':'deployment_scale_down_during_active_native_tts','ok':True,'elapsed_after_scale_s':time.monotonic()-started,'results':results}
    finally:
        for task in tasks:
            if not task.done(): task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        for forward in forwards:
            forward.terminate(); await asyncio.to_thread(forward.wait)
        await asyncio.to_thread(subprocess.run,K+['scale','deployment/voice-tts','--replicas=2'],check=True,stdout=subprocess.DEVNULL)
    out=ROOT/'.local/results/drain';out.mkdir(parents=True,exist_ok=True); (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': asyncio.run(main())
