"""Opt-in microbatch admission; cancelled native work retains its capacity."""
from __future__ import annotations

import asyncio
import time
from collections import Counter, deque
from dataclasses import dataclass
from concurrent.futures.process import BrokenProcessPool

from .runtime import integer_setting
from .workers import CALL_TIMING, WorkerBusy, WorkerPool, _timed_batch_work


@dataclass(eq=False)
class Job:
    command: str
    payload: object
    entered: float
    deadline: float
    started: asyncio.Future
    result: asyncio.Future
    active: bool = False


class BatchWorkerPool(WorkerPool):
    def __init__(self, kind, pending, size, wait_ms):
        super().__init__(kind, pending)
        self.batch_size = size
        self.wait_s = wait_ms / 1000
        self.jobs = deque()
        self.wake = asyncio.Event()
        self.dispatchers = []
        self.batch_sizes = Counter()
        self.model_batch_sizes = Counter()
        self.active_slots = set()
        self.closing = False

    async def warm(self):
        await super().warm()
        self.dispatchers = [asyncio.create_task(self._dispatch(slot))
                            for slot in range(len(self.executors))]

    def _release(self, job):
        self.admitted -= 1
        self.pending.release()

    def _expire(self, job):
        self.rejected += 1
        self.rejection_reasons['deadline'] += 1
        self.telemetry.observe('expired_queue_wait', time.monotonic() - job.entered,
                               'stt' if self.kind == 'asr' else self.kind)
        if not job.started.done():
            job.started.set_exception(WorkerBusy(f'{self.kind} queue deadline exceeded', self.kind))
        self._release(job)

    async def call(self, command, payload=None, timeout=30, queue_budget=None):
        entered = time.monotonic()
        self.offered += 1
        if self.closing:
            raise WorkerBusy(f'{self.kind} worker is closing', self.kind)
        if self.pending.locked():
            self.rejected += 1
            self.rejection_reasons['full'] += 1
            raise WorkerBusy(f'{self.kind} queue is full', self.kind)
        await self.pending.acquire()
        self.admitted += 1
        loop = asyncio.get_running_loop()
        deadline = entered + (queue_budget if queue_budget is not None else timeout)
        job = Job(command, payload, entered, deadline, loop.create_future(), loop.create_future())
        self.jobs.append(job)
        self.wake.set()
        try:
            async with asyncio.timeout_at(entered + timeout):
                try:
                    queue_wait = await asyncio.wait_for(asyncio.shield(job.started),
                                                        max(0., deadline - time.monotonic()))
                except TimeoutError as exc:
                    # Dispatch checks the same deadline before starting native work.
                    if job in self.jobs:
                        self.jobs.remove(job)
                        self._expire(job)
                    # Retrieve any exception set by the expiration path.
                    if job.started.done() and not job.started.cancelled():
                        job.started.exception()
                    raise WorkerBusy(f'{self.kind} queue deadline exceeded', self.kind) from exc
                result, inference = await asyncio.shield(job.result)
                CALL_TIMING.set((queue_wait, inference))
                return result
        finally:
            if job in self.jobs:
                self.jobs.remove(job)
                self._release(job)
            # A cancelled/expired caller must never release active native occupancy.
            job.result.cancel()
            if not job.started.done():
                job.started.cancel()

    async def _dispatch(self, slot):
        stage = 'stt' if self.kind == 'asr' else self.kind
        while not self.closing:
            while not self.jobs:
                self.wake.clear()
                await self.wake.wait()
            # Collection time counts towards the actual queue deadline.
            remaining = max(0., min(job.deadline for job in self.jobs) - time.monotonic())
            if len(self.jobs) < self.batch_size:
                await asyncio.sleep(min(self.wait_s, remaining))
            batch = []
            while self.jobs and len(batch) < self.batch_size:
                job = self.jobs.popleft()
                if job.result.cancelled():
                    self._release(job)
                elif time.monotonic() >= job.deadline:
                    self._expire(job)
                else:
                    batch.append(job)
            if not batch:
                continue
            started = time.monotonic()
            for job in batch:
                job.active = True
                wait = started - job.entered
                job.started.set_result(wait)
                self.telemetry.observe('queue_wait', wait, stage)
            self.running += len(batch)
            self.active_slots.add(slot)
            try:
                future = asyncio.get_running_loop().run_in_executor(
                    self.executors[slot], _timed_batch_work, batch[0].command,
                    [job.payload for job in batch])
                # The dispatcher owns this future, independently of all callers.
                results, wall, cpu, groups = await asyncio.shield(future)
                if len(results) != len(batch):
                    raise RuntimeError('native batch result count mismatch')
                elapsed = time.monotonic() - started
                self.batch_sizes[len(batch)] += 1
                for name, sizes in groups.items():
                    for size in sizes:
                        self.model_batch_sizes[name, size] += 1
                self.estimate = .2 * (elapsed / len(batch)) + .8 * self.estimate
                self.telemetry.observe('native_cpu', cpu, stage)
                for job, result in zip(batch, results):
                    self.telemetry.observe('inference', elapsed, stage)
                    if not job.result.done():
                        job.result.set_result((result, elapsed))
            except Exception as exc:
                if isinstance(exc, BrokenProcessPool):
                    self.healthy = False
                for job in batch:
                    if not job.result.done():
                        job.result.set_exception(exc)
            finally:
                self.active_slots.discard(slot)
                self.running -= len(batch)
                for job in batch:
                    self._release(job)

    def close(self):
        self.closing = True
        while self.jobs:
            job = self.jobs.popleft()
            if not job.started.done():
                job.started.set_exception(WorkerBusy(f'{self.kind} worker is closing', self.kind))
            self._release(job)
        # Active dispatchers keep ownership until their native futures finish.
        for slot, task in enumerate(self.dispatchers):
            if slot not in self.active_slots:
                task.cancel()
        super().close()


def make_worker_pool(kind, pending):
    if kind not in ('asr', 'tts'):
        return WorkerPool(kind, pending)
    prefix = 'VOICE_' + kind.upper()
    size = integer_setting(prefix + '_BATCH_SIZE', 1, maximum=4)
    if size == 1:
        return WorkerPool(kind, pending)
    wait_ms = integer_setting(prefix + '_BATCH_WAIT_MS', 2, maximum=10)
    if size > pending:
        raise ValueError('batch size cannot exceed admission')
    return BatchWorkerPool(kind, pending, size, wait_ms)
