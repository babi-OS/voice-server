import asyncio
import json
from types import SimpleNamespace

import aiohttp
import pytest
from aiohttp.test_utils import TestClient, TestServer

from voice_server.server import create_app, drain

PROFILE = {
    'input': {'codec': 'pcm16', 'sample_rate': 16000, 'frame_ms': 20, 'packet_ms': 20},
    'output': {'codec': 'pcm16', 'sample_rate': 24000, 'frame_ms': 20, 'packet_ms': 20},
}


def pool():
    return SimpleNamespace(admitted=0, running=0, rejected=0)


def test_metrics_and_readiness_during_drain():
    async def run():
        svc = SimpleNamespace(vad=pool(), asr=pool(), tts=pool())
        async with TestClient(TestServer(create_app(svc))) as client:
            ws = await client.ws_connect('/v2/voice')
            assert ws._response.headers['X-Voice-Replica']
            await ws.send_json({'type': 'hello', 'version': 2, 'audio': PROFILE})
            assert (await ws.receive_json())['type'] == 'ready'
            response = await client.get('/metrics')
            text = await response.text()
            assert 'voice_connections 1\n' in text
            assert 'voice_worker_queued{kind="tts"} 0\n' in text
            svc.draining = True
            assert (await client.get('/readyz')).status == 503
            assert (await client.get('/healthz')).status == 200
            with pytest.raises(aiohttp.WSServerHandshakeError) as error:
                await client.ws_connect('/v2/voice')
            assert error.value.status == 503
            await ws.close()
            assert not svc.connections
    asyncio.run(run())


def test_drain_delivers_active_tts_before_closing_websocket():
    async def run():
        svc = SimpleNamespace(vad=pool(), asr=pool(), tts=pool())
        entered, release = asyncio.Event(), asyncio.Event()

        async def synthesize(*args, **kwargs):
            entered.set()
            await release.wait()
            return 24000, [0.1] * 480

        svc.tts.call = synthesize
        async with TestClient(TestServer(create_app(svc))) as client:
            ws = await client.ws_connect('/v2/voice')
            await ws.send_json({'type': 'hello', 'version': 2, 'audio': PROFILE})
            await ws.receive_json()
            await ws.send_json({'type': 'tts.start', 'request_id': 'tts', 'turn_id': 'turn'})
            await ws.send_json({'type': 'tts.text', 'request_id': 'tts', 'index': 0, 'text': 'Xin chào.'})
            await ws.send_json({'type': 'tts.finish', 'request_id': 'tts'})
            await entered.wait()
            draining = asyncio.create_task(drain(svc, timeout=2))
            await asyncio.sleep(.05)
            assert svc.draining and not draining.done() and not ws.closed
            release.set()
            events, audio = [], []
            while True:
                message = await ws.receive(timeout=3)
                if message.type == aiohttp.WSMsgType.TEXT:
                    events.append(json.loads(message.data))
                elif message.type == aiohttp.WSMsgType.BINARY:
                    audio.append(message.data)
                elif message.type == aiohttp.WSMsgType.CLOSE:
                    assert message.data == 1012
                    break
            assert audio and any(e['type'] == 'request.end' and e['status'] == 'completed' for e in events)
            await asyncio.wait_for(draining, 3)
            assert not svc.connections
    asyncio.run(run())
