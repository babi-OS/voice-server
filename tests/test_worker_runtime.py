import asyncio
import concurrent.futures
import threading

import pytest

from voice_server import workers
from voice_server.runtime import integer_setting, session_options


def test_runtime_settings_reject_invalid_values(monkeypatch):
    for value in ("0", "-1", "33", "many"):
        monkeypatch.setenv("VOICE_TTS_WORKERS", value)
        with pytest.raises(ValueError):
            workers.WorkerPool("tts", 4)
    monkeypatch.setenv("VOICE_TTS_WORKERS", "5")
    with pytest.raises(ValueError, match="admission"):
        workers.WorkerPool("tts", 4)
    monkeypatch.setenv("VOICE_TTS_THREADS", "0")
    with pytest.raises(ValueError):
        integer_setting("VOICE_TTS_THREADS", 2)
    monkeypatch.setenv("VOICE_ORT_SPINNING", "false")
    with pytest.raises(ValueError):
        session_options(1)


def test_onnx_thread_and_spin_options(monkeypatch):
    monkeypatch.setenv("VOICE_ORT_SPINNING", "0")
    options = session_options(1)
    assert options.intra_op_num_threads == 1
    assert options.inter_op_num_threads == 1
    assert options.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    assert options.get_session_config_entry("session.inter_op.allow_spinning") == "0"


def test_workers_warm_individually_and_keep_admission_until_native_jobs_finish(monkeypatch):
    monkeypatch.setenv("VOICE_TTS_WORKERS", "2")
    monkeypatch.setattr(workers.WorkerPool, "_new", lambda _: concurrent.futures.ThreadPoolExecutor(max_workers=1))
    release = threading.Event()
    entered = threading.Barrier(3)
    warm_threads = set()

    def work(command, payload):
        if command == "ping":
            warm_threads.add(threading.get_ident())
            return "tts"
        if payload < 2:
            entered.wait(timeout=3)
        release.wait(timeout=5)
        return payload

    monkeypatch.setattr(workers, "_work", work)

    async def run():
        pool = workers.WorkerPool("tts", 4)
        tasks = []
        try:
            await pool.warm()
            assert len(warm_threads) == 2
            tasks = [asyncio.create_task(pool.call("synthesize", i)) for i in range(4)]
            await asyncio.to_thread(entered.wait, 3)
            # Both workers execute; other calls wait in the shared async queue.
            assert pool.available.empty()
            assert pool.admitted == 4 and pool.running == 2
            tasks[0].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[0]
            assert pool.available.empty()
            with pytest.raises(workers.WorkerBusy):
                await pool.call("synthesize", 4)
            assert pool.admitted == 4 and pool.running == 2 and pool.rejected == 1
            release.set()
            assert await asyncio.gather(*tasks[1:]) == [1, 2, 3]
            # The first native job finished too, despite its cancelled caller.
            for _ in range(100):
                if pool.available.qsize() == 2:
                    break
                await asyncio.sleep(.01)
            assert pool.available.qsize() == 2
            assert pool.admitted == 0 and pool.running == 0
            assert await pool.call("synthesize", 5) == 5
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            pool.close()

    asyncio.run(run())


def test_vad_remains_one_process_for_recurrent_state(monkeypatch):
    monkeypatch.setenv("VOICE_VAD_WORKERS", "8")
    monkeypatch.setattr(workers.WorkerPool, "_new", lambda _: concurrent.futures.ThreadPoolExecutor(max_workers=1))
    pool = workers.WorkerPool("vad", 64)
    try:
        assert len(pool.executors) == 1
    finally:
        pool.close()


def test_cancel_while_waiting_for_worker_releases_only_queued_admission(monkeypatch):
    monkeypatch.setenv("VOICE_TTS_WORKERS", "1")
    monkeypatch.setattr(workers.WorkerPool, "_new", lambda _: concurrent.futures.ThreadPoolExecutor(max_workers=1))
    entered = threading.Event()
    release = threading.Event()

    def work(command, payload):
        entered.set()
        release.wait(timeout=5)
        return payload

    monkeypatch.setattr(workers, "_work", work)

    async def run():
        pool = workers.WorkerPool("tts", 2)
        tasks = []
        try:
            tasks.append(asyncio.create_task(pool.call("synthesize", 0)))
            assert await asyncio.to_thread(entered.wait, 3)
            tasks.append(asyncio.create_task(pool.call("synthesize", 1)))
            await asyncio.sleep(.02)
            tasks[1].cancel()
            with pytest.raises(asyncio.CancelledError):
                await tasks[1]
            # A replacement can be admitted, but cannot steal the active worker.
            tasks.append(asyncio.create_task(pool.call("synthesize", 2)))
            await asyncio.sleep(.02)
            assert not tasks[2].done()
            assert pool.available.empty()
            with pytest.raises(workers.WorkerBusy):
                await pool.call("synthesize", 3)
            release.set()
            assert await tasks[0] == 0
            assert await tasks[2] == 2
        finally:
            release.set()
            await asyncio.gather(*tasks, return_exceptions=True)
            pool.close()

    asyncio.run(run())
