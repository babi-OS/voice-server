import asyncio,concurrent.futures,threading,time
from types import SimpleNamespace
import pytest
from aiohttp.test_utils import TestClient,TestServer
from voice_server import workers
from voice_server.server import create_app
from voice_server.gateway import Discovery,RemotePool


def test_queue_deadline_and_cancelled_native_occupancy(monkeypatch):
    monkeypatch.setattr(workers.WorkerPool,'_new',lambda _:concurrent.futures.ThreadPoolExecutor(max_workers=1))
    release=threading.Event(); entered=threading.Event()
    def work(*args): entered.set(); release.wait(4); return 'hello'
    monkeypatch.setattr(workers,'_work',work)
    async def check():
        pool=workers.WorkerPool('asr',2); pool.estimate=.01
        active=asyncio.create_task(pool.call('decode',b'xx',queue_budget=2))
        try:
            assert await asyncio.to_thread(entered.wait,2)
            started=time.monotonic()
            with pytest.raises(workers.WorkerBusy): await pool.call('decode',b'xx',queue_budget=.08)
            assert .07<time.monotonic()-started<.3
            assert pool.admitted==pool.running==1
            active.cancel()
            with pytest.raises(asyncio.CancelledError): await active
            assert pool.admitted==pool.running==1
            release.set()
            for _ in range(100):
                if not pool.admitted: break
                await asyncio.sleep(.01)
            assert pool.admitted==0
        finally: release.set(); await asyncio.gather(active,return_exceptions=True); pool.close()
    asyncio.run(check())


def test_worker_auth_validation_binary_and_overload(monkeypatch):
    monkeypatch.setenv('VOICE_RPC_TOKEN','test-token')
    async def call(command,payload,**kwargs):
        assert kwargs['queue_budget']==2
        if payload=='busy': raise workers.WorkerBusy('full')
        return 24000,[.1]*480
    svc=SimpleNamespace(role='tts',tts=SimpleNamespace(call=call),vad=None,asr=None)
    async def check():
        async with TestClient(TestServer(create_app(svc))) as client:
            assert (await client.post('/internal/v1/tts',json={'text':'hi'})).status==401
            headers={'Authorization':'Bearer test-token'}
            assert (await client.post('/internal/v1/tts',headers=headers,json={'text':'hi','sample_rate':44100})).status==400
            response=await client.post('/internal/v1/tts',headers=headers,json={'text':'hi'})
            assert response.status==200 and len(await response.read())==960
            response=await client.post('/internal/v1/tts',headers=headers,json={'text':'busy'})
            assert response.status==429 and (await response.json())['not_started']
            assert (await client.get('/v2/voice')).status==404
    asyncio.run(check())


def test_discovery_fail_closed_and_load_score():
    d=Discovery(None,'test'); d.last=time.monotonic(); d.endpoints={'stt':['http://a','http://b'],'tts':[]}
    d.capacity={'http://a':(time.monotonic(),{'admitted':2}),'http://b':(time.monotonic(),{'admitted':0})}
    assert d.choose('stt',set())=='http://b'
    d.last-=31
    with pytest.raises(workers.WorkerBusy): d.choose('stt',set())


def test_gateway_retries_explicit_prestart_rejection_only(monkeypatch):
    monkeypatch.setenv('VOICE_RPC_TOKEN','test-token')
    from aiohttp import web,ClientSession
    async def check():
        counts=[0,0]
        async def rejected(req): counts[0]+=1; return web.json_response({'not_started':True},status=429)
        async def success(req): counts[1]+=1; return web.json_response({'text':'hello'})
        apps=[]
        for handler in (rejected,success):
            app=web.Application(); app.router.add_post('/internal/v1/stt',handler); apps.append(TestServer(app))
        for app in apps: await app.start_server()
        try:
            async with ClientSession() as session:
                d=Discovery(session,'test-token'); d.last=time.monotonic(); d.endpoints['stt']=[str(a.make_url('')).rstrip('/') for a in apps]
                d.capacity={d.endpoints['stt'][1]:(time.monotonic(),{'admitted':10})}
                pool=RemotePool('stt',d)
                assert await pool.call('decode',b'xx')=='hello'
                assert counts==[1,1]
        finally:
            for app in apps: await app.close()
    asyncio.run(check())


def test_gateway_does_not_retry_ambiguous_server_failure():
    from aiohttp import web,ClientSession,ClientResponseError
    async def check():
        calls=[]
        async def handler(req): calls.append(1); return web.json_response({'not_started':False},status=500)
        app=web.Application(); app.router.add_post('/internal/v1/stt',handler)
        async with TestServer(app) as server:
            async with ClientSession() as session:
                d=Discovery(session,'token'); d.last=time.monotonic(); d.endpoints['stt']=[str(server.make_url('')).rstrip('/')]
                with pytest.raises(ClientResponseError): await RemotePool('stt',d).call('decode',b'xx')
                assert len(calls)==1
    asyncio.run(check())


def test_cancel_queued_rpc_frees_slot_when_connection_disconnects(monkeypatch):
    from aiohttp import ClientSession
    monkeypatch.setenv('VOICE_RPC_TOKEN','token')
    monkeypatch.setattr(workers.WorkerPool,'_new',lambda _:concurrent.futures.ThreadPoolExecutor(max_workers=1))
    entered=threading.Event(); release=threading.Event()
    def work(*args): entered.set(); release.wait(5); return 'hello'
    monkeypatch.setattr(workers,'_work',work)
    async def check():
        pool=workers.WorkerPool('asr',2); pool.estimate=.01
        svc=SimpleNamespace(role='stt',asr=pool,tts=None,vad=None)
        active=asyncio.create_task(pool.call('decode',b'xx'))
        await asyncio.to_thread(entered.wait,2)
        try:
            async with TestServer(create_app(svc),handler_cancellation=True) as server:
                async with ClientSession() as session:
                    request=asyncio.create_task(session.post(server.make_url('/internal/v1/stt'),data=b'xx',headers={'Authorization':'Bearer token'}))
                    for _ in range(50):
                        if pool.admitted==2: break
                        await asyncio.sleep(.01)
                    assert pool.admitted==2
                    request.cancel(); await asyncio.gather(request,return_exceptions=True)
                    for _ in range(100):
                        if pool.admitted==1: break
                        await asyncio.sleep(.01)
                    assert pool.admitted==1 and pool.running==1
        finally: release.set(); await active; pool.close()
    asyncio.run(check())


def test_multi_gateway_race_keeps_worker_admission_bounded(monkeypatch):
    from aiohttp import ClientSession
    monkeypatch.setenv('VOICE_RPC_TOKEN','token')
    monkeypatch.setattr(workers.WorkerPool,'_new',lambda _:concurrent.futures.ThreadPoolExecutor(max_workers=1))
    release=threading.Event()
    monkeypatch.setattr(workers,'_work',lambda *args:(release.wait(3) and 'hello'))
    async def check():
        pool=workers.WorkerPool('asr',2); pool.estimate=.01
        svc=SimpleNamespace(role='stt',asr=pool,tts=None,vad=None)
        try:
            async with TestServer(create_app(svc),handler_cancellation=True) as server:
                async with ClientSession() as session:
                    futures=[asyncio.create_task(session.post(server.make_url('/internal/v1/stt'),data=b'xx',headers={'Authorization':'Bearer token'})) for _ in range(8)]
                    await asyncio.sleep(.12)
                    assert pool.admitted==2 and pool.running==1
                    release.set()
                    responses=await asyncio.gather(*futures)
                    statuses=[r.status for r in responses]
                    assert statuses.count(200)==2 and statuses.count(429)==6
                    assert pool.admitted==0
        finally: release.set(); pool.close()
    asyncio.run(check())


def test_dead_native_process_fails_readiness_and_liveness(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool
    monkeypatch.setattr(workers.WorkerPool,'_new',lambda _:concurrent.futures.ThreadPoolExecutor(max_workers=1))
    def work(*_): raise BrokenProcessPool('native process died')
    monkeypatch.setattr(workers,'_work',work)
    async def check():
        pool=workers.WorkerPool('asr',2)
        try:
            with pytest.raises(BrokenProcessPool): await pool.call('decode',b'xx')
            assert pool.admitted==0 and not pool.healthy
            svc=SimpleNamespace(role='stt',asr=pool,tts=None,vad=None)
            async with TestClient(TestServer(create_app(svc))) as client:
                assert (await client.get('/readyz')).status==503
                assert (await client.get('/healthz')).status==503
        finally: pool.close()
    asyncio.run(check())


def test_already_broken_executor_fails_before_dispatch(monkeypatch):
    from concurrent.futures.process import BrokenProcessPool
    class BrokenExecutor:
        def submit(self,*_,**kw): raise BrokenProcessPool('already dead')
        def shutdown(self,**kw): pass
    monkeypatch.setattr(workers.WorkerPool,'_new',lambda _:BrokenExecutor())
    async def check():
        pool=workers.WorkerPool('asr',2)
        try:
            with pytest.raises(BrokenProcessPool): await pool.call('decode',b'xx')
            assert pool.admitted==pool.running==0 and not pool.healthy
        finally: pool.close()
    asyncio.run(check())


def test_vad_release_waits_for_bounded_queue_instead_of_leaking_state(monkeypatch):
    from voice_server.gateway import VadThread
    released=[]
    monkeypatch.setattr(workers,'_initialize',lambda _:None)
    def work(command,payload):
        if command=='release': released.append(payload)
        return None
    monkeypatch.setattr(workers,'_work',work)
    async def check():
        pool=VadThread(); pool.queue=asyncio.Queue(maxsize=1)
        try:
            await pool.warm()
            await asyncio.gather(*(pool.call('release',str(i)) for i in range(20)))
            assert set(released)=={str(i) for i in range(20)} and pool.admitted==0
        finally: await pool.aclose()
    asyncio.run(check())


def test_gateway_readiness_requires_fresh_discovery_not_both_stages():
    async def check():
        svc=SimpleNamespace(role='gateway',vad=None,asr=None,tts=None,discovery=SimpleNamespace(last=0))
        async with TestClient(TestServer(create_app(svc))) as client:
            assert (await client.get('/readyz')).status==503
            assert (await client.get('/healthz')).status==200
            svc.discovery.last=time.monotonic()
            assert (await client.get('/readyz')).status==200
            svc.discovery.last-=31
            assert (await client.get('/readyz')).status==503
            assert (await client.get('/healthz')).status==200
    asyncio.run(check())


def test_larger_queue_accounts_for_work_ahead_and_keeps_deadline(monkeypatch):
    monkeypatch.setattr(workers.WorkerPool,'_new',lambda _:concurrent.futures.ThreadPoolExecutor(max_workers=1))
    release=threading.Event(); entered=threading.Event()
    def work(*_): entered.set(); release.wait(3); return 'hello'
    monkeypatch.setattr(workers,'_work',work)
    async def check():
        pool=workers.WorkerPool('asr',8);pool.estimate=.2
        first=asyncio.create_task(pool.call('decode',b'xx',queue_budget=2))
        tasks=[first]
        try:
            await asyncio.to_thread(entered.wait,2)
            tasks.append(asyncio.create_task(pool.call('decode',b'xx',queue_budget=.5)))
            await asyncio.sleep(.01)
            with pytest.raises(workers.WorkerBusy): await pool.call('decode',b'xx',queue_budget=.25)
            assert pool.rejection_reasons['predicted_wait']==1
            tasks[1].cancel();await asyncio.gather(tasks[1],return_exceptions=True)
            assert not pool.queued_estimates
            release.set();assert await first=='hello'
        finally:release.set();await asyncio.gather(*tasks,return_exceptions=True);pool.close()
    asyncio.run(check())


def test_discovery_uses_idle_worker_outside_a_busy_sample(monkeypatch):
    # A two-choice sample can miss both idle workers in this small fleet.
    from voice_server import gateway
    monkeypatch.setattr(gateway.random,'sample',lambda urls,n:urls[:n])
    d=Discovery(None,'test');d.last=time.monotonic()
    d.endpoints['tts']=['http://a','http://b','http://c','http://d']
    d.inflight={'http://a':2,'http://b':2,'http://c':0,'http://d':0}
    assert d.choose('tts',set()) in {'http://c','http://d'}
    assert d.choose('tts',{'http://c','http://d'}) in {'http://a','http://b'}
    d.failed={'http://c':time.monotonic()}
    assert d.choose('tts',{'http://d'}) in {'http://a','http://b'}


def test_discovery_distributes_equal_load_without_hiding_idle_workers(monkeypatch):
    import random
    from voice_server import gateway
    rng=random.Random(20261021)
    monkeypatch.setattr(gateway.random,'random',rng.random)
    d=Discovery(None,'test');d.last=time.monotonic()
    d.endpoints['tts']=['http://busy','http://a','http://b','http://c','http://d']
    d.inflight={'http://busy':1}
    selected={d.choose('tts',set()) for _ in range(200)}
    assert selected=={'http://a','http://b','http://c','http://d'}
