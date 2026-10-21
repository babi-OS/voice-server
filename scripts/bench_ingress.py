#!/usr/bin/env python3
"""Real-model local benchmark, with independent pacing and stage histograms."""
import argparse,asyncio,json,time,subprocess
from pathlib import Path
import aiohttp

from bench_voice import AUDIO,until,summarize
from bench_ingress_client import Probe

async def run(args):
    def cgroup():
        if not args.container: return None
        text=subprocess.check_output(['docker','exec',args.container,'/app/.venv/bin/python','-c',"import json,pathlib; p=pathlib.Path('/sys/fs/cgroup'); print(json.dumps({'cpu':dict(line.split() for line in (p/'cpu.stat').read_text().splitlines()),'memory_current':int((p/'memory.current').read_text()),'memory_peak':int((p/'memory.peak').read_text())}))"],text=True)
        result=json.loads(text)
        native=subprocess.check_output(['docker','exec',args.container,'/app/.venv/bin/python','-c',Path(__file__).with_name('native_sample.py').read_text()],text=True)
        result['native']=json.loads(native)
        return result
    driver_cpu_started=time.process_time()
    group_before=cgroup()
    probe=Probe(args)
    config_path=Path(args.output)/'config.json'; config=json.loads(config_path.read_text())
    config.update(method='evenly staggered fixed per-user cadence' if args.mode=='staggered' else 'synchronized waves at fixed cadence',latency_budget_s=5,warmup='native startup model warm-up; no excluded full-length client turn')
    config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n')
    base=args.url.replace('ws://','http://').removesuffix('/v2/voice')
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=512),timeout=aiohttp.ClientTimeout(total=None)) as session:
        assert await probe.health(session)
        before=await (await session.get(base+'/metrics')).text()
        async def connect():
            start=time.perf_counter();ws=None
            try:
                ws=await session.ws_connect(args.url,max_msg_size=10000)
                await ws.send_json({'type':'hello','version':2,'audio':AUDIO}); await until(ws,'ready')
                return ws,time.perf_counter()-start,None
            except Exception as exc:
                if ws is not None: await ws.close()
                return None,time.perf_counter()-start,str(exc)
        conns=await asyncio.gather(*(connect() for _ in range(args.users)))
        started=time.perf_counter()
        stop_event=asyncio.Event()
        async def client(i,ws,delay,connect_error):
            for turn in range(args.turns):
                due=started+(i*args.period/args.users+turn*args.period if args.mode=='staggered' else turn*args.period)
                delay_to_due=max(0,due-time.perf_counter())
                if args.stop_on_failure:
                    try: await asyncio.wait_for(stop_event.wait(),delay_to_due)
                    except TimeoutError: pass
                    if stop_event.is_set(): break
                else: await asyncio.sleep(delay_to_due)
                lag=max(0,time.perf_counter()-due)
                if ws is None and connect_error is None: ws,delay,connect_error=await connect()
                if connect_error is not None:
                    row={'concurrency':args.users,'wave':turn,'user':i,'ok':False,'error':connect_error,'turn_s':delay,'schedule_lag_s':lag,'overload_stage':'connection'}
                    if 'overloaded' in connect_error: row['overload_response_s']=delay
                    probe.record(row);connect_error=None
                else:
                    row=await probe.turn(ws,args.users,turn,i,delay); row['schedule_lag_s']=lag
                if not row['ok']:
                    if args.stop_on_failure: stop_event.set()
                    if ws is not None: await ws.close()
                    ws=None
            if ws is not None: await ws.close()
        try: await asyncio.gather(*(client(i,*conn) for i,conn in enumerate(conns)))
        finally: await asyncio.gather(*(ws.close() for ws,_,_ in conns if ws is not None))
        after=await (await session.get(base+'/metrics')).text()
        out=Path(args.output)
        (out/'metrics-before.txt').write_text(before); (out/'metrics-after.txt').write_text(after)
        (out/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in probe.rows))
        report={'planned_attempts':args.users*args.turns,'stopped_on_failure':stop_event.is_set(),'driver_cpu_s':time.process_time()-driver_cpu_started,'cgroup_before':group_before,'cgroup_after':cgroup(),'mode':args.mode,'users':args.users,'period_s':args.period,'turns_each':args.turns,'elapsed_s':time.perf_counter()-started,'max_schedule_lag_s':max(r['schedule_lag_s'] for r in probe.rows),**summarize(probe.rows)}
        (out/'summary.json').write_text(json.dumps(report,indent=2)+'\n'); print(json.dumps(report),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',default='ws://127.0.0.1:18765/v2/voice'); p.add_argument('--fixture',default=str(Path(__file__).resolve().parents[1]/'tests/fixtures/vi_weather_two_clauses.wav')); p.add_argument('--output',required=True)
    p.add_argument('--stop-on-failure',action='store_true');p.add_argument('--container'); p.add_argument('--users',type=int,default=8); p.add_argument('--turns',type=int,default=40); p.add_argument('--period',type=float,default=15); p.add_argument('--mode',choices=['staggered','burst'],default='staggered'); p.add_argument('--profile',default='mixed',choices=['mixed','tts']); a=p.parse_args()
    if not 1<=a.users<=512 or not 1<=a.turns<=100 or a.period<5: p.error('bounded load: users 1..512, turns 1..100, period >=5s')
    asyncio.run(run(a))
