import asyncio
import concurrent.futures
import threading
import time

import pytest

from voice_server import batching, workers


def test_batch_deadlines_cancellation_order_and_native_occupancy(monkeypatch):
    monkeypatch.setattr(workers.WorkerPool, '_new',
                        lambda _: concurrent.futures.ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(workers, '_work', lambda *_: 'tts')
    monkeypatch.setattr(workers, '_calibration', lambda: .01)
    entered, release = threading.Event(), threading.Event()
    batches = []

    def native(command, payloads):
        batches.append(payloads)
        entered.set()
        release.wait(3)
        return payloads, .1, .1, {'decoder': [len(payloads)]}

    monkeypatch.setattr(batching, '_timed_batch_work', native)

    async def run():
        pool = batching.BatchWorkerPool('tts', 4, 2, 5)
        tasks = []
        try:
            await pool.warm()
            tasks = [asyncio.create_task(pool.call('synthesize', i, queue_budget=.2))
                     for i in range(2)]
            assert await asyncio.to_thread(entered.wait, 1)
            assert batches == [[0, 1]]
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            assert pool.running == 2 and pool.admitted == 2
            # Cancel a queued caller and reuse its admission immediately.
            queued = asyncio.create_task(pool.call('synthesize', 2, queue_budget=.2))
            await asyncio.sleep(.01)
            queued.cancel()
            with pytest.raises(asyncio.CancelledError):
                await queued
            assert pool.admitted == 2
            start = time.monotonic()
            with pytest.raises(workers.WorkerBusy):
                await pool.call('synthesize', 3, queue_budget=.04)
            assert time.monotonic() - start < .09
            assert pool.admitted == 2 and pool.running == 2
            assert batches == [[0, 1]]  # Expired jobs never reach native code.
            release.set()
            assert await tasks[1] == 1
            assert pool.admitted == pool.running == 0
            assert await pool.call('synthesize', 4, queue_budget=.2) == 4
            assert pool.batch_sizes == {2: 1, 1: 1}
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            pool.close()
            await asyncio.gather(*pool.dispatchers, return_exceptions=True)

    asyncio.run(run())


def test_batch_failure_releases_every_item_without_replay(monkeypatch):
    monkeypatch.setattr(workers.WorkerPool, '_new',
                        lambda _: concurrent.futures.ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(workers, '_work', lambda *_: 'asr')
    monkeypatch.setattr(workers, '_calibration', lambda: .01)
    calls = []

    def native(*args):
        calls.append(args)
        raise RuntimeError('native failure')

    monkeypatch.setattr(batching, '_timed_batch_work', native)

    async def run():
        pool = batching.BatchWorkerPool('asr', 4, 4, 2)
        try:
            await pool.warm()
            results = await asyncio.gather(*(pool.call('decode', i, queue_budget=.2)
                                             for i in range(4)), return_exceptions=True)
            assert all(isinstance(r, RuntimeError) for r in results)
            assert len(calls) == 1
            assert pool.running == pool.admitted == 0
        finally:
            pool.close()
            await asyncio.gather(*pool.dispatchers, return_exceptions=True)

    asyncio.run(run())


def test_batch_config_is_opt_in_and_bounded(monkeypatch):
    monkeypatch.setattr(workers.WorkerPool, '_new',
                        lambda _: concurrent.futures.ThreadPoolExecutor(max_workers=1))
    pool = batching.make_worker_pool('tts', 4)
    assert type(pool) is workers.WorkerPool
    pool.close()
    for value in ('0', '5', 'many'):
        monkeypatch.setenv('VOICE_TTS_BATCH_SIZE', value)
        with pytest.raises(ValueError):
            batching.make_worker_pool('tts', 4)


def test_close_rejects_queued_jobs_but_retains_running_batch(monkeypatch):
    monkeypatch.setattr(workers.WorkerPool, '_new',
                        lambda _: concurrent.futures.ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(workers, '_work', lambda *_: 'tts')
    monkeypatch.setattr(workers, '_calibration', lambda: .01)
    entered, release = threading.Event(), threading.Event()

    def native(command, payloads):
        entered.set()
        release.wait(3)
        return payloads, .1, .1, {}

    monkeypatch.setattr(batching, '_timed_batch_work', native)

    async def run():
        pool = batching.BatchWorkerPool('tts', 4, 2, 2)
        tasks = []
        try:
            await pool.warm()
            tasks = [asyncio.create_task(pool.call('synthesize', i, queue_budget=2))
                     for i in range(2)]
            assert await asyncio.to_thread(entered.wait, 1)
            queued = [asyncio.create_task(pool.call('synthesize', i, queue_budget=2))
                      for i in (2, 3)]
            tasks += queued
            await asyncio.sleep(.01)
            with pytest.raises(workers.WorkerBusy):
                await pool.call('synthesize', 4, queue_budget=2)
            pool.close()
            rejected = await asyncio.gather(*queued, return_exceptions=True)
            assert all(isinstance(r, workers.WorkerBusy) for r in rejected)
            assert pool.running == pool.admitted == 2
            release.set()
            assert await asyncio.gather(*tasks[:2]) == [0, 1]
            await asyncio.gather(*pool.dispatchers)
            assert pool.running == pool.admitted == 0
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            pool.close()

    asyncio.run(run())
