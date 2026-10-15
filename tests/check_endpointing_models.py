"""Opt-in real-model endpointing replay; downloads public models on first run."""

import asyncio
import json
import time
import wave
from pathlib import Path
from types import SimpleNamespace

from aiohttp import ClientSession, web

from voice_server.codec import encode_packet, make_encoder
from voice_server.protocol import pack_audio, validate_hello
from voice_server.server import create_app
from voice_server.workers import WorkerPool

ROOT = Path(__file__).resolve().parents[1]


async def replay(ws, pcm, duration, profile, options, request):
    input_profile = validate_hello(
        {"type": "hello", "version": 2, "audio": profile}
    ).input
    encoder = make_encoder(input_profile) if input_profile.codec == "opus" else None
    await ws.send_json(
        {"type": "stt.start", "request_id": request, "turn_id": "probe", **options}
    )
    ready = await ws.receive_json()
    assert ready["type"] == "stt.ready"
    stream = ready["stream_id"]
    start = time.monotonic()
    result = {
        "codec": input_profile.codec,
        "options": options,
        "input_seconds": duration,
    }
    events = []

    async def upload():
        tail = 0 if options.get("end_silence_ms") == 0 else 48000
        for seq, offset in enumerate(range(0, len(pcm) + tail, 640)):
            await asyncio.sleep(max(0, start + seq * 0.02 - time.monotonic()))
            frame = pcm[offset : offset + 640].ljust(640, b"\0")
            payload = encode_packet(frame, input_profile, encoder) if encoder else frame
            await ws.send_bytes(pack_audio(stream, seq, payload))
        await ws.send_json({"type": "stt.finish", "request_id": request})

    task = asyncio.create_task(upload())
    try:
        while True:
            event = await asyncio.wait_for(ws.receive_json(), 30)
            events.append(event["type"])
            if event["type"] == "stt.speech_end":
                task.cancel()
            if event["type"] == "stt.final":
                result.update(
                    transcript=event["text"],
                    post_clip_s=time.monotonic() - start - duration,
                )
            if event["type"] == "request.end":
                assert event["status"] == "completed"
                break
            if event["type"] == "error":
                assert event["code"] == "late_packet", event
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    result["events"] = events
    print(json.dumps(result, ensure_ascii=False), flush=True)
    text = result["transcript"].upper()
    if options.get("end_silence_ms") == 512:
        assert "MANG Ô" not in text
    else:
        assert "MANG Ô" in text and "SÀI GÒN" in text
    return result


async def main():
    service = SimpleNamespace(
        vad=WorkerPool("vad", 64), asr=WorkerPool("asr", 8), tts=None
    )
    runner = None
    try:
        await asyncio.gather(service.vad.warm(), service.asr.warm())
        runner = web.AppRunner(create_app(service))
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]
        with wave.open(
            str(ROOT / "tests/fixtures/vi_weather_two_clauses.wav")
        ) as source:
            pcm = source.readframes(source.getnframes())
            duration = source.getnframes() / source.getframerate()
        results = []
        async with ClientSession() as client:
            for codec in ("pcm16", "opus"):
                profile = {
                    "input": {
                        "codec": codec,
                        "sample_rate": 16000,
                        "frame_ms": 20,
                        "packet_ms": 20,
                    },
                    "output": {
                        "codec": "pcm16",
                        "sample_rate": 24000,
                        "frame_ms": 20,
                        "packet_ms": 20,
                    },
                }
                if codec == "opus":
                    profile["input"].update(
                        max_packet_bytes=1275,
                        bitrate=24000,
                        complexity=3,
                        application="restricted_lowdelay",
                    )
                async with client.ws_connect(f"http://127.0.0.1:{port}/v2/voice") as ws:
                    await ws.send_json(
                        {"type": "hello", "version": 2, "audio": profile}
                    )
                    assert (await ws.receive_json())["type"] == "ready"
                    for n, options in enumerate(
                        ({"end_silence_ms": 512}, {}, {"end_silence_ms": 0})
                    ):
                        results.append(
                            await replay(
                                ws, pcm, duration, profile, options, f"{codec}-{n}"
                            )
                        )
        output = ROOT / ".local"
        output.mkdir(exist_ok=True)
        (output / "real-model-results.json").write_text(
            json.dumps(results, ensure_ascii=False, indent=2) + "\n"
        )
    finally:
        if runner:
            await runner.cleanup()
        service.vad.close()
        service.asr.close()


if __name__ == "__main__":
    asyncio.run(main())
