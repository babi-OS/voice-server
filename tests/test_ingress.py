import asyncio
import json
import time
from types import SimpleNamespace

import pytest
from aiohttp.test_utils import TestClient, TestServer
from voice_server.server import Connection, create_app
from voice_server.protocol import pack_audio, validate_hello
from voice_server.workers import WorkerBusy
from voice_server.gateway import Discovery, RemotePool

PROFILE = {'input': {'codec':'pcm16','sample_rate':16000,'frame_ms':20,'packet_ms':20},
           'output': {'codec':'pcm16','sample_rate':24000,'frame_ms':20,'packet_ms':20}}


def test_socket_reader_rejects_stale_audio_without_waiting_for_vad():
    async def check():
        blocked = asyncio.Event()
        feeds = []
        decoded = []
        async def vad(command, payload=None, **kwargs):
            if command == 'feed':
                feeds.append(1)
                await blocked.wait()
                return 1.
        async def asr(*args, **kwargs):
            decoded.append(1)
            return 'hello'
        svc = SimpleNamespace(role='gateway',vad=SimpleNamespace(call=vad),asr=SimpleNamespace(call=asr),tts=None)
        async with TestClient(TestServer(create_app(svc))) as client:
            ws = await client.ws_connect('/v2/voice')
            await ws.send_json({'type':'hello','version':2,'audio':PROFILE})
            assert (await ws.receive_json())['type']=='ready'
            await ws.send_json({'type':'stt.start','request_id':'r','turn_id':'t','end_silence_ms':0})
            sid = (await ws.receive_json())['stream_id']
            started=time.monotonic()
            for seq in range(20):
                await ws.send_bytes(pack_audio(sid,seq,b'\x01\x00'*320))
            await ws.send_json({'type':'stt.finish','request_id':'r'})
            msg=await ws.receive_json(timeout=1.5)
            assert msg['type']=='error' and msg['code']=='overloaded' and msg['stage']=='ingress'
            assert time.monotonic()-started<1.1
            assert feeds and not decoded
            await ws.close()
            blocked.set()
            for _ in range(50):
                if not client.app['ingress'].streams: break
                await asyncio.sleep(.01)
            assert client.app['ingress'].pending==0
    asyncio.run(check())


def test_ingress_full_is_bounded_and_no_audio_replayed(monkeypatch):
    monkeypatch.setenv('VOICE_INGRESS_ITEMS','2')
    async def check():
        from voice_server.ingress import Ingress, IngressLimiter
        from aiohttp import WSMsgType
        class Conn:
            def __init__(self): self.ws=self;self.calls=[];self.events=[];self.stt_task=self.tts_task=None
            async def packet(self, data): self.calls.append(data);await asyncio.sleep(1)
            async def control(self,*args,**kwargs): pass
            async def send(self, kind, **fields): self.events.append(fields)
            async def close(self, **kwargs): pass
        c=Conn();limiter=IngressLimiter();ingress=Ingress(c,limiter)
        for _ in range(10): ingress.offer(WSMsgType.BINARY,b'xx')
        await asyncio.sleep(.01)
        assert ingress.closed and limiter.pending<=2
        assert c.events[0]['code']=='overloaded'
        await ingress.close()
        assert limiter.pending==0 and not c.calls
    asyncio.run(check())


def test_ingress_age_reduces_submitted_stt_queue_budget():
    async def check():
        calls=[]
        class Socket:
            async def send_str(self, data): pass
        async def call(*args,**kw): calls.append(kw['queue_budget']);return 'hello'
        svc=SimpleNamespace(role='gateway',asr=SimpleNamespace(kind='stt',call=call),vad=None,tts=None)
        c=Connection(Socket(),svc)
        c.stt='r';c.speaking=True;c.utterance.extend(b'\x01\x00'*8000)
        await c.finish_stt(received_at=time.monotonic()-.3)
        await c.stt_task
        assert len(calls)==1 and 1.5<calls[0]<1.71
        with pytest.raises(WorkerBusy):
            await c.stage_call(svc.asr,'decode',b'xx',timeout=90,received_at=time.monotonic()-2.1)
        assert len(calls)==1
    asyncio.run(check())


def test_remote_retry_does_not_refresh_remaining_queue_deadline():
    async def check():
        from aiohttp import web, ClientSession
        calls=[]
        async def handler(req):
            calls.append(int(req.headers['X-Queue-Budget-Ms']))
            await asyncio.sleep(.1)
            return web.json_response({'not_started':True},status=429)
        app=web.Application();app.router.add_post('/internal/v1/stt',handler)
        async with TestServer(app) as server:
            async with ClientSession() as session:
                d=Discovery(session,'test');d.last=time.monotonic();d.endpoints['stt']=[str(server.make_url('')).rstrip('/')]
                with pytest.raises(WorkerBusy): await RemotePool('stt',d).call('decode',b'xx',queue_budget=.08)
                assert len(calls)==1 and 0<calls[0]<=80
    asyncio.run(check())


def test_failed_ingress_watchdog_fails_liveness():
    async def check():
        svc=SimpleNamespace(role='gateway',vad=None,asr=None,tts=None)
        async with TestClient(TestServer(create_app(svc))) as client:
            svc.ingress.task.cancel()
            await asyncio.gather(svc.ingress.task,return_exceptions=True)
            assert (await client.get('/healthz')).status==503
    asyncio.run(check())
