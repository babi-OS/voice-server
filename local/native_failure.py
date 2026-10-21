"""Kill only a disposable local Pod's native process and verify kubelet recovery."""
import asyncio,json,subprocess,time,wave
from pathlib import Path
import aiohttp
from common import wait_ready_port
ROOT=Path(__file__).resolve().parents[1]; K=['kubectl','--kubeconfig',str(ROOT/'.local/kubeconfig'),'--context','k3d-voice-local','-n','voice-split-bench']; NAME='voice-native-probe'
def k(*args): return subprocess.check_output(K+list(args),text=True)
async def main():
    template=json.loads(k('get','deployment','voice-stt','-o','json'))['spec']['template']
    pod={'apiVersion':'v1','kind':'Pod','metadata':{'name':NAME,'namespace':'voice-split-bench','labels':{'app':NAME}},'spec':template['spec']}
    subprocess.run(K+['apply','-f','-'],input=json.dumps(pod),text=True,check=True,stdout=subprocess.DEVNULL)
    forward=None; report={}
    try:
        await asyncio.to_thread(k,'wait','pod/'+NAME,'--for=condition=Ready','--timeout=180s')
        forward=subprocess.Popen(K+['port-forward','pod/'+NAME,'18775:8765','--address','127.0.0.1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        async with aiohttp.ClientSession() as session:
            await wait_ready_port(session,18775,forward)
            code="import os,signal,pathlib; pids=[int(p.name) for p in pathlib.Path('/proc').iterdir() if p.name.isdigit() and int(p.name)!=os.getpid() and b'multiprocessing.spawn' in (p/'cmdline').read_bytes()]; assert len(pids)==1,pids; os.kill(pids[0],signal.SIGKILL)"
            started=time.monotonic(); await asyncio.to_thread(k,'exec',NAME,'--','/app/.venv/bin/python','-c',code)
            for _ in range(40):
                async with session.get('http://127.0.0.1:18775/readyz') as response: status=response.status
                if status==503: break
                await asyncio.sleep(.1)
            assert status==503
            async with session.get('http://127.0.0.1:18775/healthz') as response: assert response.status==503
            for _ in range(90):
                current=json.loads(await asyncio.to_thread(k,'get','pod',NAME,'-o','json'))
                containers=current['status'].get('containerStatuses',[])
                ready=any(c['type']=='Ready' and c['status']=='True' for c in current['status'].get('conditions',[]))
                if containers and containers[0]['restartCount']>=1 and ready: break
                await asyncio.sleep(1)
            else: raise RuntimeError('kubelet did not recover native worker')
            forward.terminate(); await asyncio.to_thread(forward.wait)
            forward=subprocess.Popen(K+['port-forward','pod/'+NAME,'18775:8765','--address','127.0.0.1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            await wait_ready_port(session,18775,forward)
            with wave.open(str(ROOT/'tests/fixtures/vi_weather_two_clauses.wav')) as f: pcm=f.readframes(f.getnframes())
            headers={'Authorization':'Bearer '+(ROOT/'.local/rpc-token').read_text().strip()}
            async with session.post('http://127.0.0.1:18775/internal/v1/stt',data=pcm,headers=headers) as response:
                text=(await response.json())['text']; assert response.status==200 and text
            report={'ok':True,'test':'native_process_death_liveness_restart','recovery_s':time.monotonic()-started,'restart_count':containers[0]['restartCount'],'post_recovery_transcript':text}
    finally:
        if forward:
            forward.terminate(); await asyncio.to_thread(forward.wait)
        subprocess.run(K+['delete','pod',NAME,'--wait=false'],check=True,stdout=subprocess.DEVNULL)
    out=ROOT/'.local/results/native-failure'; out.mkdir(parents=True,exist_ok=True); (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report,indent=2))
if __name__=='__main__': asyncio.run(main())
