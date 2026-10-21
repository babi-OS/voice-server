"""Observe overload while streaming input, as a real duplex voice client does."""
import asyncio
import json
import time
import aiohttp
from bench_voice import Probe as OriginalProbe


class DuplexSocket:
    def __init__(self, ws):
        self.ws=ws;self.messages=asyncio.Queue();self.error=None;self.error_at=None
        self.submitted_at=None;self.submitted_stage=None
        self.reader=asyncio.create_task(self.read())

    async def read(self):
        try:
            while True:
                message=await self.ws.receive()
                if message.type==aiohttp.WSMsgType.TEXT:
                    d=json.loads(message.data)
                    if d.get('code')=='overloaded' or d.get('result',{}).get('code')=='overloaded':
                        self.error=d;self.error_at=time.perf_counter()
                await self.messages.put(message)
                if message.type in (aiohttp.WSMsgType.CLOSE,aiohttp.WSMsgType.CLOSED,aiohttp.WSMsgType.ERROR):break
        except asyncio.CancelledError:raise

    def check(self):
        if self.error:
            raise RuntimeError('server_overloaded:'+json.dumps(self.error))

    async def send_json(self, d):
        self.check()
        if d.get('type') in ('stt.finish','tts.text'):
            self.submitted_at=time.perf_counter();self.submitted_stage='stt' if d['type']=='stt.finish' else 'tts'
        await self.ws.send_json(d)

    async def send_bytes(self, data):
        self.check();await self.ws.send_bytes(data)

    async def receive(self, timeout=135):
        return await asyncio.wait_for(self.messages.get(),timeout)

    async def close(self):
        self.reader.cancel();await asyncio.gather(self.reader,return_exceptions=True)


class Probe(OriginalProbe):
    async def turn(self, ws, *args):
        duplex=DuplexSocket(ws)
        try:
            row=await super().turn(duplex,*args)
            if duplex.error:
                row['overload_detected_during_streaming']=duplex.submitted_at is None
                row['overload_event']=duplex.error
                if duplex.submitted_at is not None:
                    row['overload_response_s']=max(0.,duplex.error_at-duplex.submitted_at)
                    row['overload_stage']=duplex.submitted_stage
            return row
        finally:await duplex.close()
