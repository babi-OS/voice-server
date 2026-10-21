"""Explicit, bounded loopback benchmark of a running voice server; never automatic."""
import argparse
import asyncio
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
import urllib.request

import aiohttp
from .benchmark_client import Probe, AUDIO, until, summarize


def snapshot():
    root = Path('/sys/fs/cgroup')
    result = {'time': time.time(), 'driver_cpu_s': time.process_time()}
    for name in ('cpu.stat','cpu.max','memory.current','memory.peak','memory.max','memory.events','cpu.pressure'):
        try:
            result[name] = (root / name).read_text().strip()
        except OSError:
            pass
    try:
        result['cpu_model'] = next(line.split(':',1)[1].strip() for line in
            Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name'))
    except (OSError, StopIteration):
        pass
    result['models'] = {key: os.getenv(key) for key in
        ('ASR_REVISION','MATCHA_REVISION','VAD_REVISION','VOICE_ASR_THREADS','VOICE_TTS_THREADS',
         'VOICE_ASR_BATCH_SIZE','VOICE_TTS_BATCH_SIZE','VOICE_ASR_BATCH_WAIT_MS','VOICE_TTS_BATCH_WAIT_MS')}
    if directory := os.getenv('MATCHA_CPU_DIR'):
        try:
            raw = (Path(directory) / 'model.json').read_bytes()
            metadata = json.loads(raw)
            result['local_tts_model'] = {'manifest_sha256': hashlib.sha256(raw).hexdigest(),
                                         'graphs': metadata.get('files'),
                                         'batch_variant': metadata.get('batch_variant')}
        except (OSError, ValueError):
            result['local_tts_model'] = {'error': 'manifest unavailable'}
    return result


def stage_metrics():
    result = {}
    for port in (8765,19001,19002,19003,19004,19101,19102,19103,19104):
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/metrics',timeout=.5) as r:
                result[str(port)] = r.read().decode()
        except OSError:
            pass
    return result


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--fixture',default=str(Path(__file__).parent/'fixtures/vi_weather_two_clauses.wav'))
    p.add_argument('--users',type=int,default=1)
    p.add_argument('--turns',type=int,default=2)
    p.add_argument('--period',type=float,default=15)
    args = p.parse_args(argv)
    if not 1 <= args.users <= 32 or not 1 <= args.turns <= 40 or not 15 <= args.period <= 30 or args.turns*args.period > 600:
        p.error('bounded load: users 1..32, turns 1..40, period 15..30s, scheduled duration <=600s')
    # No remote URL option: this tool must report the deployment it runs inside.
    args.url = 'ws://127.0.0.1:8765/v2/voice'
    args.profile = 'mixed'
    return args


async def measure(args):
    probe = Probe(args)
    before = snapshot()
    metrics_before = await asyncio.to_thread(stage_metrics)
    started = time.monotonic()
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=64),
                                    timeout=aiohttp.ClientTimeout(total=None)) as session:
        if not await probe.health(session):
            raise RuntimeError('local voice server is not healthy')
        async def client(i):
            async with session.ws_connect(args.url,max_msg_size=10000) as ws:
                await ws.send_json({'type':'hello','version':2,'audio':AUDIO})
                await until(ws,'ready')
                for turn in range(args.turns):
                    due = started + i*args.period/args.users + turn*args.period
                    await asyncio.sleep(max(0,due-time.monotonic()))
                    lag = max(0,time.monotonic()-due)
                    row = await probe.turn(ws,args.users,turn,i,0)
                    row['schedule_lag_s'] = lag
                    if not row['ok']:
                        break
        async with asyncio.timeout(args.turns*args.period+150):
            await asyncio.gather(*(client(i) for i in range(args.users)))
    elapsed = time.monotonic()-started
    after = snapshot()
    metrics_after = await asyncio.to_thread(stage_metrics)
    report = {'schema':1,'scope':'loopback real Opus/STT/TTS; benchmark driver shares serving cgroup',
              'fixture_sha256':hashlib.sha256(Path(args.fixture).read_bytes()).hexdigest(),
              'users':args.users,'turns_each':args.turns,'period_s':args.period,
              'planned_attempts':args.users*args.turns,'elapsed_s':elapsed,
              'before':before,'after':after,**summarize(probe.rows),
              'max_schedule_lag_s':max((r['schedule_lag_s'] for r in probe.rows),default=0),
              'metrics_before':metrics_before,'metrics_after':metrics_after}
    if 'cpu.stat' in before and 'cpu.stat' in after:
        start = dict(line.split() for line in before['cpu.stat'].splitlines())
        end = dict(line.split() for line in after['cpu.stat'].splitlines())
        cpu_s = (int(end['usage_usec'])-int(start['usage_usec']))/1e6
        driver_s = after['driver_cpu_s']-before['driver_cpu_s']
        report.update(cgroup_cpu_s=cpu_s,driver_cpu_s=driver_s,
                      mean_cpu_cores=cpu_s/elapsed)
    out = Path(args.output)
    (out/'requests.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in probe.rows))
    (out/'summary.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in
                     ('metrics_before','metrics_after','before','after')},ensure_ascii=False),flush=True)
    return report


def main():
    args = arguments()
    # One opt-in benchmark per container, with no public load-generation route.
    with open('/tmp/voice-benchmark.lock','a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('another built-in benchmark is already running')
        asyncio.run(measure(args))


if __name__ == '__main__':
    main()
