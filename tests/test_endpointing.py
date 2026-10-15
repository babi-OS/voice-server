import asyncio
import json
from types import SimpleNamespace

import pytest

from voice_server.protocol import ProtocolError, end_silence_ms, validate_hello
from voice_server.server import MAX_UTTERANCE_BYTES, VAD_WINDOW_BYTES, Connection
from voice_server.workers import WorkerBusy

PROFILE = {
    "input": {"codec": "pcm16", "sample_rate": 16000, "frame_ms": 20, "packet_ms": 20},
    "output": {"codec": "pcm16", "sample_rate": 24000, "frame_ms": 20, "packet_ms": 20},
}


class Socket:
    def __init__(self):
        self.events = []

    async def send_str(self, text):
        self.events.append(json.loads(text))


class Pool:
    def __init__(self):
        self.calls = []
        self.block = None
        self.entered = asyncio.Event()
        self.destroyed = asyncio.Event()
        self.failure = None

    async def call(self, command, payload=None, **kwargs):
        self.calls.append((command, payload))
        if self.failure:
            raise self.failure
        if command in ("decode", "synthesize"):
            self.entered.set()
            try:
                if self.block:
                    await self.block.wait()
                if command == "decode":
                    return "HÔM NAY Ở SÀI GÒN BAO NHIÊU ĐỘ CÓ NÊN MANG Ô KHÔNG"
                return 24000, [0.0] * 480
            finally:
                self.destroyed.set()


def connection():
    svc = SimpleNamespace(vad=Pool(), asr=Pool(), tts=Pool())
    c = Connection(Socket(), svc)
    c.audio = validate_hello({"type": "hello", "version": 2, "audio": PROFILE})
    return c


async def start(c, request="r", **options):
    await c.start_stt({"request_id": request, "turn_id": "turn", **options})


async def windows(c, count, probability, marker=1):
    for _ in range(count):
        await c.vad_window(bytes([marker]) * VAD_WINDOW_BYTES, probability)


@pytest.mark.parametrize("value", [None, True, False, 255, 3001, -1, 1000.0, "1000"])
def test_invalid_endpoint_setting(value):
    with pytest.raises(ProtocolError):
        end_silence_ms({"end_silence_ms": value})


@pytest.mark.parametrize("value", [0, 256, 687, 1000, 3000])
def test_valid_endpoint_setting(value):
    assert end_silence_ms({"end_silence_ms": value}) == value


def test_default_keeps_both_clauses_across_672ms_pause():
    async def go():
        c = connection()
        await start(c)
        await windows(c, 10, 1, 1)
        await windows(c, 21, 0, 0)
        assert c.stt == "r" and not c.svc.asr.calls
        await windows(c, 10, 1, 2)
        await windows(c, 31, 0, 0)
        assert c.stt == "r"
        await windows(c, 1, 0, 0)
        await c.stt_task
        decoded = c.svc.asr.calls[0][1]
        assert bytes([1]) * VAD_WINDOW_BYTES in decoded
        assert bytes([2]) * VAD_WINDOW_BYTES in decoded
        assert [e["type"] for e in c.ws.events].count("stt.speech_end") == 1
        assert [e["type"] for e in c.ws.events].count("stt.final") == 1
        assert c.ws.events[-1]["status"] == "completed"
        await c.close()

    asyncio.run(go())


def test_explicit_endpoint_rounds_up_to_complete_vad_window():
    async def go():
        c = connection()
        await start(c, end_silence_ms=687)
        await windows(c, 10, 1)
        await windows(c, 21, 0)
        assert c.stt == "r"  # 672ms is still inside the requested tolerance.
        await windows(c, 1, 0)
        await c.stt_task
        assert c.ws.events[-1]["status"] == "completed"
        await c.close()

    asyncio.run(go())


def test_client_endpointing_retains_pending_pcm_and_next_request_resets_policy():
    async def go():
        c = connection()
        await start(c, end_silence_ms=0)
        await windows(c, 10, 1)
        await windows(c, 80, 0)
        assert c.stt == "r"
        tail = b"\x33" * 640
        c.pending.extend(tail)
        await c.control(json.dumps({"type": "stt.finish", "request_id": "r"}))
        await c.stt_task
        assert c.svc.asr.calls[0][1].endswith(tail)
        events = len(c.ws.events)
        await c.control(json.dumps({"type": "stt.finish", "request_id": "r"}))
        assert len(c.ws.events) == events
        await start(c, "next")
        assert c.end_silence_windows == 32
        await c.close()

    asyncio.run(go())


def test_client_endpointing_keeps_the_utterance_limit():
    async def go():
        c = connection()
        await start(c, end_silence_ms=0)
        c.speaking = True
        c.utterance.extend(b"\0" * MAX_UTTERANCE_BYTES)
        c.pending.extend(b"\0" * 2)
        await c.finish_stt()
        assert c.ws.events[-1]["result"]["code"] == "utterance_too_long"
        assert not c.svc.asr.calls
        await c.close()

    asyncio.run(go())


def test_failed_vad_admission_rolls_back_and_invalid_settings_do_not_reset_worker():
    async def go():
        c = connection()
        with pytest.raises(ProtocolError):
            await start(c, end_silence_ms=True)
        assert not c.svc.vad.calls and c.stt is None
        c.svc.vad.failure = WorkerBusy("full")
        with pytest.raises(WorkerBusy):
            await start(c)
        assert c.stt is None and not c.ws.events
        c.svc.vad.failure = None
        await start(c, "next")
        assert c.stt == "next"
        await c.close()

    asyncio.run(go())


def test_finish_during_decode_is_idempotent_and_cancel_drains_before_reuse():
    async def go():
        c = connection()
        c.svc.asr.block = asyncio.Event()
        await start(c)
        await windows(c, 10, 1)
        await c.finish_stt()
        await c.svc.asr.entered.wait()
        await c.control(json.dumps({"type": "stt.finish", "request_id": "r"}))
        with pytest.raises(ProtocolError, match="active"):
            await c.start_tts({"request_id": "tts", "turn_id": "turn"})
        await c.cancel("r")
        assert c.svc.asr.destroyed.is_set() and c.stt_task is None
        assert [e["status"] for e in c.ws.events if e["type"] == "request.end"] == [
            "cancelled"
        ]
        await start(c, "next")
        await asyncio.sleep(0)
        assert c.stt == "next" and c.stt_task is None
        await c.close()

    asyncio.run(go())


def test_tts_cancel_and_disconnect_drain_tasks_before_release():
    async def go():
        c = connection()
        c.svc.tts.block = asyncio.Event()
        await c.start_tts({"request_id": "tts", "turn_id": "turn"})
        await c.tts_text({"request_id": "tts", "index": 0, "text": "Xin chào"})
        await c.svc.tts.entered.wait()
        await c.cancel("tts")
        assert c.svc.tts.destroyed.is_set() and c.tts_task is None
        c.svc.asr.block = asyncio.Event()
        await start(c, "stt")
        await windows(c, 10, 1)
        await c.finish_stt()
        await c.svc.asr.entered.wait()
        await c.close()
        assert c.svc.asr.destroyed.is_set() and c.stt_task is None
        assert c.svc.vad.calls[-1][0] == "release"

    asyncio.run(go())
