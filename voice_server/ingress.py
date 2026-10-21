"""Bounded socket ingestion, independent from ordered codec/VAD processing.

Queue age is measured at socket receipt. A shared watchdog can shed a stale
stream while its consumer is waiting for a codec/native operation. It never
releases that operation's native capacity early.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from dataclasses import dataclass
from aiohttp import WSMsgType
from .runtime import integer_setting
from .protocol import ProtocolError
from .workers import WorkerBusy
from .telemetry import Telemetry

# Leave time for the downstream queue and for reporting overload to the peer.
LOG=logging.getLogger('voice_server.ingress')
MAX_AGE_S = .5
REPORT_TIMEOUT_S = .2


@dataclass
class Message:
    kind: WSMsgType
    data: str | bytes
    received: float


class IngressLimiter:
    def __init__(self):
        self.limit = integer_setting('VOICE_INGRESS_GLOBAL_ITEMS', 1024, maximum=8192)
        self.pending = 0
        self.streams = set()
        self.rejections = Counter()
        self.telemetry = Telemetry()
        self.task = None

    async def start(self):
        self.task = asyncio.create_task(self.monitor())

    async def monitor(self):
        while True:
            await asyncio.sleep(.025)
            now = time.monotonic()
            for stream in tuple(self.streams):
                oldest = stream.oldest
                if oldest is not None and now - oldest > MAX_AGE_S:
                    stream.shed('queue_age', oldest)

    async def close(self):
        if self.task:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
        await asyncio.gather(*(s.close() for s in tuple(self.streams)))


class Ingress:
    def __init__(self, connection, limiter):
        self.connection = connection
        self.limiter = limiter
        self.queue = asyncio.Queue(maxsize=integer_setting('VOICE_INGRESS_ITEMS', 32, maximum=128))
        self.current = None
        self.first_queued = None
        self.closed = False
        self.abort_task = None
        limiter.streams.add(self)
        self.consumer = asyncio.create_task(self.consume())

    @property
    def oldest(self):
        return self.current.received if self.current is not None else self.first_queued

    def offer(self, kind, data):
        if self.closed:
            return
        now = time.monotonic()
        if self.queue.full() or self.limiter.pending >= self.limiter.limit:
            self.shed('queue_full', self.oldest or now)
            return
        item = Message(kind, data, now)
        self.queue.put_nowait(item)
        if self.first_queued is None:
            self.first_queued = now
        self.limiter.pending += 1

    def shed(self, reason, received=None):
        if self.closed:
            return
        self.closed = True
        self.limiter.rejections[reason] += 1
        self.consumer.cancel()
        while not self.queue.empty():
            self.queue.get_nowait()
            self.limiter.pending-=1
        for task in (self.connection.stt_task,self.connection.tts_task):
            if task: task.cancel()
        self.abort_task = asyncio.create_task(self.reject(received or time.monotonic(), reason))

    async def reject(self, received, reason):
        c = self.connection
        try:
            # Report before waiting for native cancellation/cleanup. A slow peer
            # cannot hold admission indefinitely through the send lock.
            async with asyncio.timeout(REPORT_TIMEOUT_S):
                await c.send('error', code='internal_error' if reason=='processing_failed' else 'overloaded', stage='ingress',
                             message='audio ingress unavailable; reconnect', retry_after_ms=1000)
        except (TimeoutError, ConnectionError, RuntimeError):
            pass
        finally:
            self.limiter.telemetry.observe('processing_error_response' if reason=='processing_failed' else 'overload_response', time.monotonic()-received, 'ingress')
            try:
                async with asyncio.timeout(REPORT_TIMEOUT_S):
                    await c.ws.close(code=1011 if reason=='processing_failed' else 1013, message=b'audio ingress unavailable')
            except (TimeoutError, ConnectionError, RuntimeError):
                # aiohttp's handler teardown closes the transport after this task.
                pass

    async def consume(self):
        c = self.connection
        try:
            while True:
                item = await self.queue.get()
                self.limiter.pending -= 1
                self.current = item
                # The active item predates all queued messages. There is no
                # await between finishing it and taking the next queued item.
                self.first_queued = None
                self.limiter.telemetry.observe('queue_wait', time.monotonic()-item.received, 'ingress')
                if time.monotonic()-item.received > MAX_AGE_S:
                    self.shed('queue_age', item.received)
                    return
                try:
                    if item.kind == WSMsgType.TEXT:
                        await c.control(item.data, received_at=item.received)
                    else:
                        await c.packet(item.data)
                except (ProtocolError, WorkerBusy) as exc:
                    if getattr(exc, 'code', None) == 'overloaded':
                        self.shed('stage_pressure', item.received)
                        return
                    await c.send('error', code=getattr(exc, 'code', 'busy'), message=str(exc))
                except Exception:
                    LOG.exception('voice ingress processing failed')
                    self.shed('processing_failed', item.received)
                    return
                finally:
                    self.current = None
        except asyncio.CancelledError:
            raise

    async def close(self):
        self.closed = True
        self.limiter.streams.discard(self)
        self.consumer.cancel()
        await asyncio.gather(self.consumer, return_exceptions=True)
        while not self.queue.empty():
            self.queue.get_nowait()
            self.limiter.pending -= 1
        if self.abort_task:
            await asyncio.gather(self.abort_task, return_exceptions=True)
