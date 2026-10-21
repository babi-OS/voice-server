"""Local-only recovery and drain checks; run after capacity benchmarks."""
import asyncio,json,subprocess,sys,time
from pathlib import Path
import aiohttp
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from bench_voice import Probe,AUDIO,until
from types import SimpleNamespace
K=['kubectl','--kubeconfig',str(ROOT/'.local/kubeconfig'),'--context','k3d-voice-local','-n','voice-split-bench']
def k(*args): return subprocess.check_output(K+list(args),text=True)
async def wait_ready(count=2,timeout=180):
    for _ in range(timeout):
        pods=json.loads(await asyncio.to_thread(k,'get','pods','-l','app=voice-tts','-o','json'))['items']
        ready=[p for p in pods if not p['metadata'].get('deletionTimestamp') and any(c['type']=='Ready' and c['status']=='True' for c in p.get('status',{}).get('conditions',[]))]
        if len(ready)>=count: return ready
        await asyncio.sleep(1)
    raise RuntimeError('workers did not recover')
async def main():
    out=ROOT/'.local/results/failures'; out.mkdir(parents=True,exist_ok=True); report=[]
    args=SimpleNamespace(output=str(out),fixture=str(ROOT/'tests/fixtures/vi_weather_two_clauses.wav'),url='ws://127.0.0.1:18765/v2/voice',profile='mixed')
    probe=Probe(args)
    async with aiohttp.ClientSession() as session:
        ws=await session.ws_connect(args.url); await ws.send_json({'type':'hello','version':2,'audio':AUDIO}); await until(ws,'ready')
        row=await probe.turn(ws,1,0,0,0); assert row['ok']; report.append({'test':'before_failure','ok':True})
        ready=await wait_ready(); victim=ready[0]['metadata']['name']
        await asyncio.to_thread(k,'delete','pod',victim,'--wait=false')
        started=time.monotonic(); ready=await wait_ready()
        row=await probe.turn(ws,1,1,0,0); assert row['ok']
        report.append({'test':'worker_replacement_existing_websocket','ok':True,'recovery_s':time.monotonic()-started,'replacement_pods':[p['metadata']['name'] for p in ready]})
        # Pending capacity must not appear in discovery. Use an impossible node
        # selector on a separate canary, not a real cluster resource exhaustion.
        deployment=json.loads(k('get','deployment','voice-tts','-o','json'))
        deployment['metadata']={'name':'voice-pending','namespace':'voice-split-bench'}
        deployment['spec']['replicas']=1
        deployment['spec']['selector']={'matchLabels':{'app':'voice-pending'}}
        deployment['spec']['template']['metadata']={'labels':{'app':'voice-pending'}}
        deployment['spec']['template']['spec']['nodeSelector']={'voice-test-node':'does-not-exist'}
        await asyncio.to_thread(subprocess.run,K+['apply','-f','-'],input=json.dumps(deployment),text=True,check=True,stdout=subprocess.DEVNULL)
        await asyncio.sleep(3)
        pending=json.loads(k('get','pods','-l','app=voice-pending','-o','json'))['items']
        assert pending and all(p['status']['phase']=='Pending' for p in pending)
        report.append({'test':'unschedulable_worker_pending','ok':True,'note':'synthetic placement failure; not physical resource exhaustion'})
        await asyncio.to_thread(k,'delete','deployment','voice-pending','--wait=false')
        # Stop one containerized agent; the other remains a local recovery path.
        node=ready[0]['spec']['nodeName']
        await ws.close() # Gateway may be on the node being stopped; no session resume is promised.
        started=time.monotonic()
        try:
            await asyncio.to_thread(subprocess.run,['docker','stop','-t','10',node],check=True,stdout=subprocess.DEVNULL)
            await asyncio.sleep(8)
            # Route may be on the failed node; use a temporary explicit local
            # port-forward to a surviving ready gateway for the failure smoke.
            gateways=json.loads(k('get','pods','-l','app=voice-gateway','-o','json'))['items']
            survivor=next(p for p in gateways if p['spec']['nodeName']!=node and not p['metadata'].get('deletionTimestamp'))
            pf=subprocess.Popen(K+['port-forward','pod/'+survivor['metadata']['name'],'18769:8765','--address','127.0.0.1'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            try:
                await asyncio.sleep(2)
                async with session.ws_connect('ws://127.0.0.1:18769/v2/voice') as live:
                    await live.send_json({'type':'hello','version':2,'audio':AUDIO}); await until(live,'ready')
                    attempts=[]
                    for trial in range(5):
                        row=await probe.turn(live,1,2+trial,0,0); attempts.append(row)
                        if row['ok']: break
                        await asyncio.sleep(1)
                    assert row['ok'],attempts
                report.append({'test':'agent_stop_surviving_gateway','ok':True,'elapsed_s':time.monotonic()-started,'failed_turns_before_recovery':sum(not r['ok'] for r in attempts),'note':'single-host container failure; reconnect via surviving gateway port-forward'})
            finally: pf.terminate(); await asyncio.to_thread(pf.wait)
        finally:
            await asyncio.to_thread(subprocess.run,['docker','start',node],check=True,stdout=subprocess.DEVNULL)
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
if __name__=='__main__': asyncio.run(main())
